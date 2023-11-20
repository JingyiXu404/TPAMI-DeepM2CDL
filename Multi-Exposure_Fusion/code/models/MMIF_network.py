from typing import Any, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch import autograd
import models.basicblock as B
import torch.nn.functional as F
import numpy as np
from math import ceil
from .utils import *
# iteration version with beta

class HeadNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], out_nc: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_nc+1,
                      nc_x[0],
                      d_size,
                      padding=(d_size - 1) // 2,
                      bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], nc_x[0], 3, padding=1, bias=False))

        self.head_d = torch.zeros(1, out_nc, nc_x[0], d_size, d_size)

    def forward(self, y: Any, sigma: Any) -> Tuple[Tensor, Tensor]:
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        d = self.head_d.repeat(y.size(0), 1, 1, 1, 1).to(y.device)
        return x, d
class MSHeadNet(nn.Module):
    def __init__(self, c_iter: int, in_nc: int, nc_x: List[int], out_nc: int, d_size: List[int]):
        super(MSHeadNet, self).__init__()
        self.c_iter = c_iter
        self.head_list: nn.ModuleList = nn.ModuleList()
        for i in range(c_iter):
            self.head_list.append(HeadNet(in_nc, nc_x, out_nc, d_size[i]))

    def forward(self, y: Any, sigma: Any) -> Tuple[List[Tensor], List[Tensor]]:
        x_list = []
        d_list = []
        for i in range(self.c_iter):
            x,d = self.head_list[i](y,sigma)
            x_list.append(x)
            d_list.append(d)
        return x_list,d_list
class Stage(nn.Module):
    # for single input, one stage
    def __init__(self, in_nc: int, nc_x: List[int], nc_d: List[int],out_nc: int, nb: int):
        super(Stage, self).__init__()

        self.net_x = NetX(in_nc=in_nc, nc_x=nc_x, nb=nb)
        self.solve_fft = SolveFFT()

        self.net_d = NetD(nc_d=nc_d, out_nc=out_nc)
        self.solve_ls = SolveLS()

    def forward(self, x: Tensor, d: Tensor, y: Tensor, Y: Tensor,
                alpha_x: Tensor, beta_x: Tensor, alpha_d: float, beta_d: float,
                reg: float):
        # Solve X
        X, D = self.rfft_xd(x, d)
        size_x = np.array(list(x.shape[-2:]))
        x = self.solve_fft(X, D, Y, alpha_x, size_x)
        beta_x = (1 / beta_x.sqrt()).repeat(1, 1, x.size(2), x.size(3))
        x = self.net_x(torch.cat([x, beta_x], dim=1))

        # Solve D
        if self.net_d is not None:
            d = self.solve_ls(x.unsqueeze(1), d, y.unsqueeze(2), alpha_d, reg)
            beda_d = (1 / beta_d.sqrt()).repeat(1, 1, d.size(3), d.size(4))
            size_d = [d.size(1), d.size(2)]
            d = d.view(d.size(0), d.size(1) * d.size(2), d.size(3), d.size(4))
            d = self.net_d(torch.cat([d, beda_d], dim=1))
            d = d.view(d.size(0), size_d[0], size_d[1], d.size(2), d.size(3))

        return x, d

    def rfft_xd(self, x: Tensor, d: Tensor):
        X = torch.rfft(x, 2).unsqueeze(1)
        D = p2o(d, x.shape[-2:])
        return X, D

class DCD(nn.Module):
    # n_iter stages of one modality
    def __init__(self,n_iter: int = 1,nc_x: List[int] = [64, 128, 256, 512],out_nc: int = 1,nb: int = 1,**kargs):
        super(DCD, self).__init__()
        self.n_iter = n_iter
        self.hypa_list: nn.ModuleList = nn.ModuleList()
        self.body = Stage(in_nc=nc_x[0]+1, nc_x=nc_x, nc_d=nc_x, out_nc=out_nc, nb=nb)  # Ji iteration of X or Y
        for _ in range(n_iter):
            self.hypa_list.append(HyPaNet(in_nc=1, out_nc=4))
    def forward(self,x,X,a,d,sigma):
        for i in range(self.n_iter):
            hypas = self.hypa_list[i](sigma)
            alpha_x = hypas[:, 0].unsqueeze(-1)
            beta_x = hypas[:, 1].unsqueeze(-1)
            alpha_d = hypas[:, 2].unsqueeze(-1)
            beta_d = hypas[:, 3].unsqueeze(-1)
            a, d = self.body(a, d, x, X, alpha_x, beta_x, alpha_d, beta_d,0.001)
        return a,d
class MSDCD(nn.Module):
    # c_iter scales and n_iter stages of one modality
    def __init__(self,n_iter: int = 1,c_iter: int = 1,nc_x: List[int] = [64, 128, 256, 512],out_nc: int = 1,nb: int = 1,**kargs):
        super(MSDCD, self).__init__()
        self.n_iter = n_iter  #number of iterations
        self.c_iter = c_iter  #number of scales
        self.scale_list: nn.ModuleList = nn.ModuleList()
        self.tail = TailNet()
        for _ in range(c_iter):
            self.scale = DCD(n_iter=n_iter, nc_x=nc_x, out_nc=out_nc, nb=nb)
            self.scale_list.append(self.scale)

    def forward(self,x,X,a,d,sigma):
        a_list = []
        d_list = []
        h, w = x.size()[-2:]
        for i in range(self.c_iter):
            out_a, out_d = self.scale_list[i](x,X,a[i],d[i],sigma)
            a_list.append(out_a)
            d_list.append(out_d)
            x = x - self.tail(out_a, out_d)[..., :h, :w]
            X = torch.rfft(x, 2)
            X = X.unsqueeze(2)

        return a_list,d_list
class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 65,
                 nc_x: List[int] = [64, 128, 256, 512],
                 nb: int = 4):
        super(NetX, self).__init__()

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(in_nc, in_nc, bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(in_nc, nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], nc_x[0], bias=False, mode='C')

    def forward(self, x):
        x1 = x
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1[:, :-1, :, :])
        return x
class SolveFFT(nn.Module):
    def __init__(self):
        super(SolveFFT, self).__init__()

    def forward(self, X: Tensor, D: Tensor, Y: Tensor, alpha: Tensor,
                x_size: np.ndarray):
        """
            X: N, 1, C_in, H, W, 2
            D: N, C_out, C_in, H, W, 2
            Y: N, C_out, 1, H, W, 2
            alpha: N, 1, 1, 1
        """
        alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

        _D = cconj(D)
        Z = cmul(Y, D) + alpha * X

        factor1 = Z / alpha

        numerator = cmul(_D, Z).sum(2, keepdim=True)
        denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                           alpha.squeeze(-1)**2)
        factor2 = cmul(D, cdiv(numerator, denominator))
        X = (factor1 - factor2).mean(1)
        return torch.irfft(X, 2, signal_sizes=list(x_size))
class NetD(nn.Module):
    def __init__(self, nc_d: List[int] = [16], out_nc: int = 1):
        super(NetD, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0]+1, out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = x
        x = self.relu(self.mlp(x))
        x = self.relu(self.mlp2(x))
        x = self.mlp3(x) + x1[:, :-1, :, :]
        return x

class UFB(nn.Module):
    def __init__(self, nc_x: List[int] = [64, 128, 256, 512]):
        super(UFB, self).__init__()
        self.conv_feature = nn.Sequential(nn.Conv2d(nc_x[0] * 2, nc_x[0], 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(nc_x[0], nc_x[0], 3, padding=1, bias=False))
    def forward(self, x):
        x = self.conv_feature(x)
        return x
class UDB(nn.Module):
    def __init__(self, nc_d: List[int] = [16], out_nc: int=1):
        super(UDB, self).__init__()
        self.conv_dictionary = nn.Sequential(nn.Conv2d(out_nc * nc_d[0]*2, out_nc * nc_d[0], 3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1, bias=False))
    def forward(self, x):
        x = self.conv_dictionary(x)
        return x
class FB(nn.Module):
    def __init__(self, nc_x: List[int] = [64, 128, 256, 512], nc_d: List[int] = [16], out_nc: int = 1):
        super(FB, self).__init__()

        self.update_feature_x = UFB(nc_x)
        self.update_feature_y = UFB(nc_x)
        self.update_dictionary_x = UDB(nc_d, out_nc)
        self.update_dictionary_y = UDB(nc_d, out_nc)

    def forward(self, u, v, dx, dy):
        '''
        :param u: input features of x-mode
        :param v: input features of y-mode
        :param dx: input dictionaries of x-mode
        :param dy: input dictionaries of y-mode
        :return f, g, b, e: joint features f and dictionaries b to match f of x-mode, joint features g and dictionaries e to match g of y-mode
        '''
        a = torch.cat([u, v], dim=1)
        f = self.update_feature_x(a)
        g = self.update_feature_y(a)
        size_b = [dx.size(1), dx.size(2)]
        dx = dx.view(dx.size(0), dx.size(1) * dx.size(2), dx.size(3), dx.size(4))
        size_e = [dy.size(1), dy.size(2)]
        dy = dy.view(dy.size(0), dy.size(1) * dy.size(2), dy.size(3), dy.size(4))
        d = torch.cat([dx, dy], dim=1)
        b = self.update_dictionary_x(d)
        e = self.update_dictionary_y(d)
        b = b.view(b.size(0), size_b[0], size_b[1], b.size(2), b.size(3))
        e = e.view(e.size(0), size_e[0], size_e[1], e.size(2), e.size(3))
        return f, g, b, e
class MSFB(nn.Module):
    def __init__(self, c_iter: int = 1, nc_x: List[int]=[64, 128, 256, 512], nc_d: List[int] = [16], out_nc: int=1):
        super(MSFB, self).__init__()
        self.c_iter = c_iter
        self.FB_list: nn.ModuleList = nn.ModuleList()
        for _ in range(c_iter):
            self.FB_list.append(FB(nc_x, nc_d, out_nc))

    def forward(self, u, v, dx, dy):
        '''
        :param u=[u1,u2,...,uc]: input features of x-mode, ui means the i-th scale
        :param v=[v1,v2,...,vc]: input features of y-mode
        :param dx=[dx1,dx2,...,dxc]: input dictionaries of x-mode
        :param dy=[dy1,dy2,...,dyc]: input dictionaries of y-mode
        :return f, b: joint features of x-mode and dictionaries to match f
        '''
        f_list = []
        g_list = []
        b_list = []
        e_list = []
        for i in range(self.c_iter):
            f, g, b, e = self.FB_list[i](u[i],v[i],dx[i],dy[i])
            f_list.append(f)
            g_list.append(g)
            b_list.append(b)
            e_list.append(e)
        return f_list,g_list,b_list,e_list
class CholeskySolve(autograd.Function):
    @staticmethod
    def forward(ctx, Q, P):
        L = torch.cholesky(Q)
        D = torch.cholesky_solve(P, L)  # D = Q-1 @ P
        ctx.save_for_backward(L, D)
        return D

    @staticmethod
    def backward(ctx, dLdD):
        L, D = ctx.saved_tensors
        dLdP = torch.cholesky_solve(dLdD, L)
        dLdQ = -dLdP.matmul(D.transpose(-2, -1))

        return dLdQ, dLdP
class SolveLS(nn.Module):
    def __init__(self):
        super(SolveLS, self).__init__()

        self.cholesky_solve = CholeskySolve.apply

    def forward(self, x, d, y, alpha, reg):
        """
            x: N, 1, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            y: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
            reg: float
        """
        C_in = x.shape[2]
        d_size = d.shape[-1]

        xtx_raw = self.cal_xtx(x, d_size)  # N, C_in, C_in, d_size, d_size
        xtx_unfold = F.unfold(
            xtx_raw.view(
                xtx_raw.size(0) * xtx_raw.size(1), xtx_raw.size(2),
                xtx_raw.size(3), xtx_raw.size(4)), d_size)
        xtx_unfold = xtx_unfold.view(xtx_raw.size(0), xtx_raw.size(1),
                                     xtx_unfold.size(1), xtx_unfold.size(2))

        xtx = xtx_unfold.view(xtx_unfold.size(0), xtx_unfold.size(1),
                              xtx_unfold.size(1), -1, xtx_unfold.size(3))
        xtx.copy_(xtx[:, :, :, torch.arange(xtx.size(3) - 1, -1, -1), ...])
        xtx = xtx.view(xtx.size(0), -1, xtx.size(-1))  # TODO
        index = torch.arange(
            (C_in * d_size)**2).view(C_in, C_in, d_size,
                                     d_size).permute(0, 2, 3, 1).reshape(-1)
        xtx.copy_(xtx[:, index, :])  # TODO
        xtx = xtx.view(xtx.size(0), d_size**2 * C_in, -1)

        xty = self.cal_xty(x, y, d_size)
        xty = xty.reshape(xty.size(0), xty.size(1), -1).permute(0, 2, 1)

        # reg
        alpha = alpha * x.size(3) * x.size(4) * reg / (d_size**2 * d.size(2))
        xtx[:, range(len(xtx[0])), range(len(
            xtx[0]))] = xtx[:, range(len(xtx[0])),
                            range(len(xtx[0]))] + alpha.squeeze(-1).squeeze(-1)
        xty += alpha.squeeze(-1) * d.reshape(d.size(0), d.size(1), -1).permute(
            0, 2, 1)

        # solve
        try:
            d = self.cholesky_solve(xtx, xty).view(d.size(0), C_in, d_size,
                                                   d_size, d.size(1)).permute(
                                                       0, 4, 1, 2, 3)
        except RuntimeError:
            pass

        return d

    def cal_xtx(self, x, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
        """
        padding = d_size - 1
        xtx = conv3d(x,
                     x.view(x.size(0), x.size(2), 1, 1, x.size(3), x.size(4)),
                     padding,
                     sample_wise=True)

        return xtx

    def cal_xty(self, x, y, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
            y: N, C_out, 1, H, W
        """
        padding = (d_size - 1) // 2

        xty = conv3d(x, y.unsqueeze(3), padding, sample_wise=True)
        return xty
class TailNet(nn.Module):
    def __init__(self):
        super(TailNet, self).__init__()

    def forward(self, x, d):
        y = conv2d(F.pad(x, [(d.size(-1) - 1) // 2,] * 4, mode='circular'),d,sample_wise=True)

        return y
class HyPaNet(nn.Module):
    def __init__(
            self,
            in_nc: int = 1,
            nc: int = 256,
            out_nc: int = 8,
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x
class DCDicL(nn.Module):
    def __init__(self,
                 n_iter_x: int = 1,
                 n_iter_y: int = 1,
                 rnn_iter: int = 1,
                 c_iter: int = 1,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256, 512],
                 out_nc: int = 1,
                 nb: int = 1,
                 d_size: List[int] = [5,3],
                 **kargs):
        super(DCDicL, self).__init__()
        self.rnn_iter = rnn_iter
        self.headx = MSHeadNet(c_iter, in_nc, nc_x, out_nc, d_size)
        self.heady = MSHeadNet(c_iter, in_nc, nc_x, out_nc, d_size)

        self.tail = TailNet()

        self.body_listx: nn.ModuleList = nn.ModuleList() # J1 iteration of X
        self.body_listy: nn.ModuleList = nn.ModuleList() # J2 iteration of Y
        self.FB_list: nn.ModuleList = nn.ModuleList()
        for _ in range(self.rnn_iter):
            self.FB_list.append(MSFB(c_iter, nc_x, nc_x, out_nc))
            self.body_listx.append(MSDCD(n_iter=n_iter_x, c_iter=c_iter, nc_x=nc_x, out_nc=out_nc, nb=nb))
            self.body_listy.append(MSDCD(n_iter=n_iter_y, c_iter=c_iter, nc_x=nc_x, out_nc=out_nc, nb=nb))



    def forward(self, x,y, sigma,sigmay):
        # padding
        h, w = y.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)
        x = F.pad(x, [0, paddingRight, 0, paddingBottom], mode='circular')
        y = F.pad(y, [0, paddingRight, 0, paddingBottom], mode='circular')

        # prepare X,Y
        X = torch.rfft(x, 2)
        X = X.unsqueeze(2)
        Y = torch.rfft(y, 2)
        Y = Y.unsqueeze(2)

        # get initial feature map and dictionary from head_net
        ax, dx = self.headx(x, sigma)
        ay, dy = self.heady(y, sigma)


        predsx = []
        predsh = []
        predsy = []
        AX = []
        AY = []
        FA = []
        GA = []
        BX = []
        EY = []
        DX = []
        DY = []
        for i in range(self.rnn_iter):
            ax, dx = self.body_listx[i](x, X, ax, dx,sigma)
            ay, dy = self.body_listy[i](y, Y, ay, dy,sigma) # get new items from the beginning for Y
            AX.append(ax)
            AY.append(ay)
            DX.append(dx)
            DY.append(dy)

            fax, fay, b, e = self.FB_list[i](ax, ay, dx, dy)
            FA.append(fax)
            GA.append(fay)
            BX.append(b)
            EY.append(e)
            # update ax,ay and dx,dy
            ax = fax
            ay = fay
            dx = b
            dy = e
            for j in range(len(ax)):
                if j == 0:
                    rx = self.tail(ax[j], dx[j])
                    ry = self.tail(ay[j], dy[j])
                else:
                    rx = rx + self.tail(ax[j], dx[j])
                    ry = ry + self.tail(ay[j], dy[j])
            # reconstruct for loss

            predx = rx[..., :h, :w]
            predsx.append(predx)
            predy = ry[..., :h, :w]
            predsy.append(predy)
            pred_h = 0.5 * predx[..., :h, :w] + 0.5 * predy[..., :h, :w]
            predsh.append(pred_h)
            # update x,X,y,Y
            x = rx
            X = torch.rfft(x, 2)
            X = X.unsqueeze(2)
            y = ry
            Y = torch.rfft(y, 2)
            Y = Y.unsqueeze(2)



        if self.training:
            return predsh,predsx,predsy
        else:
            return predsh,predsx,predsy,DX,DY,AX,AY,BX,EY,FA,GA