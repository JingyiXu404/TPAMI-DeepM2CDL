import copy
from logging import Logger

from models.MMIR_network import DCDicL
import os
from glob import glob
from typing import Any, Dict, List, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.optim import Adam, lr_scheduler
import matplotlib.pyplot as plt
from models.MMIR_DN_select_network import select_network
from utils import utils_image as util
import freeze
import matplotlib
import warnings
matplotlib.use('agg')
warnings.filterwarnings(action='ignore')
class Model:
    def __init__(self, opt: Dict[str, Any],lam1:float,lam2:float,lam3:float):
        self.opt = opt
        self.opt_train = self.opt['train']
        self.opt_test = self.opt['test']
        self.x_iter = opt['netG']['x_iter']
        self.y_iter = opt['netG']['y_iter']
        self.rnn_iter = opt['netG']['rnn_iter']
        self.c_iter = opt['netG']['c_iter']
        self.channels = opt['netG']['nc_x'][0]
        self.save_dir: str = opt['path']['models']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.type = opt['netG']['type']

        self.net = select_network(opt).to(self.device)
        # print(self.net)
        # print('# network parameters:', sum(param.numel() for param in self.net.parameters()) / 1e6, 'M')
        self.net = DataParallel(self.net)
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.schedulers = []
        self.log_dict = {}
        self.metrics = {}
    def init(self):

        self.load()

        self.net.train()

        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()

    def load(self):
        load_path = self.opt['path']['pretrained_netG']
        if load_path is not None:
            print('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net)

    def load_network(self, load_path: str, network: Union[nn.DataParallel, DCDicL]):
        if isinstance(network, nn.DataParallel):
            network: DCDicL = network.module

        #load pretrained model
        for j in range(self.c_iter):
            network.headx.head_list[j].load_state_dict(torch.load(load_path + 'headx' + '_c' + str(j) + '.pth'), strict=True)
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                network.heady_list[i].head_list[j].load_state_dict(torch.load(load_path + 'heady_' + str(i) + '_c' + str(j) + '.pth'), strict=True)
                network.RB_list[i].RB_list[j].conv_feature.load_state_dict(torch.load(load_path + 'RBF_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                network.RB_list[i].RB_list[j].conv_dictionary.load_state_dict(torch.load(load_path + 'RBD_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                network.body_listx[i].scale_list[j].body.net_x.load_state_dict(torch.load(load_path + 'ax_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                network.body_listx[i].scale_list[j].body.net_d.load_state_dict(torch.load(load_path + 'dx_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                network.body_listy[i].scale_list[j].body.net_x.load_state_dict(torch.load(load_path + 'ay_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                network.body_listy[i].scale_list[j].body.net_d.load_state_dict(torch.load(load_path + 'dy_r' + str(i)+'_c'+str(j) + '.pth'),strict=True)
                for m in range(self.x_iter):
                    state_dict_hypa_x = torch.load(load_path + 'hypax_r' + str(i) + '_c' + str(j) + '_i' + str(m) + '.pth')
                    if self.opt['train']['reload_broadcast']:
                        state_dict_hypa_v2_x = copy.deepcopy(state_dict_hypa_x)
                        for key in state_dict_hypa_x:
                            if '0.mlp' in key:  # load hypa from the first iteration
                                state_dict_hypa_v2_x[key.replace('0.mlp', 'mlp')] = state_dict_hypa_v2_x.pop(key)
                            else:
                                state_dict_hypa_v2_x.pop(key)
                        for hypa in network.body_listx[i].scale_list[j].hypa_list[m]:
                            hypa.load_state_dict(state_dict_hypa_v2_x, strict=True)
                    else:
                        network.body_listx[i].scale_list[j].hypa_list[m].load_state_dict(state_dict_hypa_x, strict=True)
                for m in range(self.y_iter):
                    state_dict_hypa_y = torch.load(load_path + 'hypay_r' + str(i) + '_c' + str(j) + '_i' + str(m) + '.pth')
                    if self.opt['train']['reload_broadcast']:
                        state_dict_hypa_v2_y = copy.deepcopy(state_dict_hypa_y)
                        for key in state_dict_hypa_y:
                            if '0.mlp' in key:  # load hypa from the first iteration
                                state_dict_hypa_v2_y[key.replace('0.mlp', 'mlp')] = state_dict_hypa_v2_y.pop(key)
                            else:
                                state_dict_hypa_v2_y.pop(key)
                        for hypa in network.body_listy[i].scale_list[j].hypa_list[m]:
                            hypa.load_state_dict(state_dict_hypa_v2_y, strict=True)
                    else:
                        network.body_listy[i].scale_list[j].hypa_list[m].load_state_dict(state_dict_hypa_y, strict=True)

    def save(self, logger: Logger):
        logger.info('Saving the model.')
        net = self.net
        if isinstance(net, nn.DataParallel):
            net = net.module

        for j in range(self.c_iter):
            self.save_network(net.headx.head_list[j], 'headx' + '_c' + str(j))
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                self.save_network(net.heady_list[i].head_list[j], 'heady_' + str(i) + '_c' +str(j))
                self.save_network(net.RB_list[i].RB_list[j].conv_feature, 'RBF_r' + str(i)+'_c'+str(j))
                self.save_network(net.RB_list[i].RB_list[j].conv_dictionary, 'RBD_r' + str(i)+'_c'+str(j))
                self.save_network(net.body_listx[i].scale_list[j].body.net_x, 'ax_r' + str(i)+'_c'+str(j))
                self.save_network(net.body_listx[i].scale_list[j].body.net_d, 'dx_r' + str(i)+'_c'+str(j))
                self.save_network(net.body_listy[i].scale_list[j].body.net_x, 'ay_r' + str(i)+'_c'+str(j))
                self.save_network(net.body_listy[i].scale_list[j].body.net_d, 'dy_r' + str(i)+'_c'+str(j))
                for m in range(self.x_iter):
                    self.save_network(net.body_listx[i].scale_list[j].hypa_list[m], 'hypax_r' + str(i) + '_c' + str(j) + '_i' + str(m))
                for m in range(self.y_iter):
                    self.save_network(net.body_listy[i].scale_list[j].hypa_list[m], 'hypay_r' + str(i) + '_c' + str(j) + '_i' + str(m))

    def save_best(self, logger: Logger):
        logger.info('Saving the model.')
        net = self.net
        if isinstance(net, nn.DataParallel):
            net = net.module
        for j in range(self.c_iter):
            self.save_network(net.headx.head_list[j], 'headx' + '_c' + str(j),best=True)
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                self.save_network(net.heady_list[i].head_list[j], 'heady_' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.RB_list[i].RB_list[j].conv_feature, 'RBF_r' + str(i)+'_c'+str(j), best=True)
                self.save_network(net.RB_list[i].RB_list[j].conv_dictionary, 'RBD_r' + str(i)+'_c'+str(j), best=True)
                self.save_network(net.body_listx[i].scale_list[j].body.net_x, 'ax_r' + str(i)+'_c'+str(j), best=True)
                self.save_network(net.body_listx[i].scale_list[j].body.net_d, 'dx_r' + str(i)+'_c'+str(j), best=True)
                self.save_network(net.body_listy[i].scale_list[j].body.net_x, 'ay_r' + str(i)+'_c'+str(j), best=True)
                self.save_network(net.body_listy[i].scale_list[j].body.net_d, 'dy_r' + str(i)+'_c'+str(j), best=True)
                for m in range(self.x_iter):
                    self.save_network(net.body_listx[i].scale_list[j].hypa_list[m], 'hypax_r' + str(i) + '_c' + str(j) + '_i' + str(m), best=True)
                for m in range(self.y_iter):
                    self.save_network(net.body_listy[i].scale_list[j].hypa_list[m], 'hypay_r' + str(i) + '_c' + str(j) + '_i' + str(m), best=True)
    def save_network(self, network, network_label,best=False):
        filename = '{}.pth'.format(network_label)
        if best:
            os.makedirs(self.save_dir+'/best/', exist_ok=True)
            save_path = os.path.join(self.save_dir+'/best/', filename)
        else:
            save_path = os.path.join(self.save_dir, filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

    def define_loss(self):
        self.lossfn = nn.L1Loss().to(self.device)

    def define_optimizer(self):
        optim_params = []
        for _, v in self.net.named_parameters():
            optim_params.append(v)
        self.optimizer = Adam(optim_params,
                              lr=self.opt_train['G_optimizer_lr'],
                              weight_decay=0)

    def define_scheduler(self):
        self.schedulers.append(
            lr_scheduler.MultiStepLR(self.optimizer,
                                     self.opt_train['G_scheduler_milestones'],
                                     self.opt_train['G_scheduler_gamma']))

    def update_learning_rate(self, n: int):
        for scheduler in self.schedulers:
            scheduler.step(n)

    @property
    def learning_rate(self) -> float:
        return self.schedulers[0].get_lr()[0]

    def feed_data(self, data: Dict[str, Any]):
        self.y = data['y'].to(self.device)
        self.y_gt = data['y_gt'].to(self.device)
        self.guide_gt=data['guide_gt'].to(self.device)
        self.sigma = data['sigma'].to(self.device)
        self.sigmay = data['sigmay'].to(self.device)
        self.path = data['path']
        self.path_guide = data['path_guide']

    def cal_multi_loss(self, preds: List[torch.Tensor],gt: torch.Tensor,lam:float) -> torch.Tensor:
        losses = None
        for i, pred in enumerate(preds):
            loss = self.lossfn(pred, gt)
            if i != len(preds) - 1:
                loss *= lam*(1 / (len(preds) - 1))
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses
    def cal_multi_loss2(self, preds: List[torch.Tensor],gt: torch.Tensor) -> torch.Tensor:
        losses = None
        for i, pred in enumerate(preds):
            loss = self.lossfn(pred, gt)
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses/(len(preds))

    def log_train(self, current_step: int, epoch: int, logger: Logger,GL:list,XL:list,XLL:list,YL:list,XH:list,name:str):
        self.epoch=epoch
        message = f'Training epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{self.learning_rate:.3e}'

        for k, v in self.log_dict.items():  # merge log information into message
            if k == 'G_loss':
                GL.append(v)
            elif k == 'X_loss':
                XL.append(v)
            elif k == 'Y_loss':
                YL.append(v)
            elif k == 'X_loss_last':
                XLL.append(v)
            elif k == 'XH_loss':
                XH.append(v)
            message += f', {k:s}: {v:.3e}'
        logger.info(message)
        fig1 = plt.figure()
        plt.subplot(3, 2, 1)
        plt.plot(GL)
        plt.ylabel('GL')
        plt.subplot(3, 2, 2)
        plt.plot(XL)
        plt.ylabel('XL')
        plt.subplot(3, 2, 3)
        plt.plot(XLL)
        plt.ylabel('XLL')
        plt.subplot(3, 2, 4)
        plt.plot(YL)
        plt.ylabel('YL')
        plt.subplot(3, 2, 5)
        plt.plot(XH)
        plt.ylabel('XH')

        plt.savefig(name)
        plt.close('all')
        return GL,XL,XLL,YL,XH
    def test(self,timedic=[]):
        self.net.eval()

        with torch.no_grad():
            y = self.y
            guide = self.guide_gt

            self.rx,self.ry,self.DX,self.DY,self.AX,self.AY,self.FA = self.net(y,guide, self.sigma,self.sigmay)

        self.prepare_visuals()

        self.net.train()

    def prepare_visuals(self):
        """ prepare visual for first sample in batch """
        self.out_dict = {}
        self.out_dict['y'] = util.tensor2uint(self.y[0].detach().float().cpu())
        self.out_dict['guide'] = util.tensor2uint(self.guide_gt[0].detach().float().cpu())
        self.out_dict['y_gt'] = util.tensor2uint(self.y_gt[0].detach().float().cpu())
        self.out_dict['path'] = self.path[0]
        for i in range(self.rnn_iter):
            self.out_dict['rx_r' + str(i)] = util.tensor2uint(self.rx[i].detach().float().cpu())
            self.out_dict['ry_r' + str(i)] = util.tensor2uint(self.ry[i].detach().float().cpu())
            for j in range(self.c_iter):
                self.out_dict['dx_r' + str(i) + '_c' + str(j)] = self.DX[i][j][0].detach().float().cpu()
                self.out_dict['dy_r' + str(i) + '_c' + str(j)] = self.DY[i][j][0].detach().float().cpu()
                for k in range(self.channels):
                    self.out_dict['ax_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.AX[i][j][0][k:k+1, ...].detach().float().cpu())
                    self.out_dict['ay_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.AY[i][j][0][k:k+1, ...].detach().float().cpu())
                    self.out_dict['fa_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.FA[i][j][0][k:k+1, ...].detach().float().cpu())

    def cal_metrics(self):
        self.metrics['psnr'] = util.calculate_psnr(self.out_dict['rx_r1'],self.out_dict['y_gt'])
        self.metrics['ssim'] = util.calculate_ssim(self.out_dict['rx_r1'],self.out_dict['y_gt'])

        return self.metrics['psnr'], self.metrics['ssim']
    def visual_features(self,img):
        result = np.zeros(img.shape, dtype=np.float32)
        cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result = np.uint8(result * 255.0)
        return result
    def save_visuals(self, tag: str):
        y_img = self.out_dict['y']
        gt_img = self.out_dict['y_gt']
        guide_img = self.out_dict['guide']
        path = self.out_dict['path']
        rx_img = []
        ry_img = []
        dx_img = []
        dy_img = []
        ax_img = []
        ay_img = []
        fa_img = []
        for i in range(self.rnn_iter):
            rx_img.append(self.out_dict['rx_r' + str(i)])
            ry_img.append(self.out_dict['ry_r' + str(i)])
            dx_img_sub = []
            dy_img_sub = []
            ax_img_sub = []
            ay_img_sub = []
            fa_img_sub = []
            for j in range(self.c_iter):
                dx_img_sub.append(self.out_dict['dx_r' + str(i) + '_c' + str(j)])
                dy_img_sub.append(self.out_dict['dy_r' + str(i) + '_c' + str(j)])
                ax_img_sub_sub = []
                ay_img_sub_sub = []
                fa_img_sub_sub = []
                for k in range(self.channels):
                    ax_img_sub_sub.append(self.out_dict['ax_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                    ay_img_sub_sub.append(self.out_dict['ay_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                    fa_img_sub_sub.append(self.out_dict['fa_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                ax_img_sub.append(ax_img_sub_sub)
                ay_img_sub.append(ay_img_sub_sub)
                fa_img_sub.append(fa_img_sub_sub)
            dx_img.append(dx_img_sub)
            dy_img.append(dy_img_sub)
            ax_img.append(ax_img_sub)
            ay_img.append(ay_img_sub)
            fa_img.append(fa_img_sub)

        img_name = os.path.splitext(os.path.basename(path))[0]
        img_dir = os.path.join(self.opt['path']['images'], img_name)
        os.makedirs(img_dir, exist_ok=True)

        old_img_path = os.path.join(img_dir, f"{img_name:s}_{tag}_*_*.png")
        old_img = glob(old_img_path)
        for img in old_img:
            os.remove(img)

        img_path = os.path.join(
            img_dir,
            f"{img_name}_{tag}_{round(self.metrics['psnr'],2)}_{round(self.metrics['ssim'],4)}.png"
        )
        util.imsave(rx_img[self.rnn_iter-1], img_path)
        # for i in range(self.rnn_iter):
        #     util.imsave(rx_img[i], img_path.replace('.png', '_x_r' + str(i + 1) + '.png'))
        #     util.imsave(ry_img[i], img_path.replace('.png', '_y_r' + str(i + 1) + '.png'))
        #     for j in range(self.c_iter):
        #         os.makedirs(os.path.join(img_dir, 'ax_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
        #         os.makedirs(os.path.join(img_dir, 'ay_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
        #         os.makedirs(os.path.join(img_dir, 'fa_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
        #         for k in range(self.channels):
        #             util.imsave(self.visual_features(ax_img[i][j][k]), os.path.join(img_dir,'ax_r'+ str(i + 1) + '_c' + str(j + 1) , str(k + 1) + '.png'))
        #             util.imsave(self.visual_features(ay_img[i][j][k]), os.path.join(img_dir,'ay_r'+ str(i + 1) + '_c' + str(j + 1) , str(k + 1) + '.png'))
        #             util.imsave(self.visual_features(fa_img[i][j][k]), os.path.join(img_dir,'fa_r'+ str(i + 1) + '_c' + str(j + 1) , str(k + 1) + '.png'))
        # util.imsave(y_img, img_path.replace('.png', '_noise.png'))
        # util.imsave(gt_img, img_path.replace('.png', '_gt.png'))
        # util.imsave(guide_img, img_path.replace('.png', '_guide.png'))
        # if self.opt['test']['visualize']:
        #     for i in range(self.rnn_iter):
        #         for j in range(self.c_iter):
        #             util.save_d(dx_img[i][j].mean(0).numpy(), img_path.replace('.png', '_dx_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))
        #             util.save_d(dy_img[i][j].mean(0).numpy(), img_path.replace('.png', '_dy_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))

    def get_freeze_rnn1(self):
        import freeze
        freeze.freeze_by_names(self.net.module, ('headx'))
        freeze.freeze_by_names(self.net.module.heady_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listx, ('0'))
        freeze.freeze_by_names(self.net.module.body_listy, ('0'))
        freeze.freeze_by_names(self.net.module.RB_list, ('0'))
    def get_freeze_rnn2(self):
        import freeze
        freeze.freeze_by_names(self.net.module, ('headx'))
        freeze.freeze_by_names(self.net.module.heady_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listx, ('0'))
        freeze.freeze_by_names(self.net.module.body_listy, ('0'))
        freeze.freeze_by_names(self.net.module.RB_list, ('0'))
    def get_freeze_rnn2_c1(self):
        import freeze
        freeze.freeze_by_names(self.net.module.heady_list[1].head_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listx[1].scale_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listy[1].scale_list, ('0'))
        # freeze.freeze_by_names(self.net.module.RB_list[1].RB_list, ('0'))
    def get_freeze_rnn2_c2(self):
        import freeze
        freeze.freeze_by_names(self.net.module.heady_list[1].head_list, ('1'))
        freeze.freeze_by_names(self.net.module.body_listx[1].scale_list, ('1'))
        freeze.freeze_by_names(self.net.module.body_listy[1].scale_list, ('1'))
        # freeze.freeze_by_names(self.net.module.RB_list[1].RB_list, ('1'))
    def train(self,i):
        # self.get_freeze_rnn1()
        # self.get_freeze_rnn2_c1()
        # self.get_freeze_rnn2_c2()
        self.optimizer.zero_grad()
        rxs, rys = self.net(self.y, self.guide_gt, self.sigma, self.sigmay)
        xh_loss = self.lossfn(rxs[-1], self.y_gt)
        x_loss = self.cal_multi_loss2(rxs, self.y_gt)
        y_loss = self.cal_multi_loss2(rys, self.guide_gt)
        x_loss_last = self.cal_multi_loss([rxs[-1]], self.y_gt, lam=1)
        loss = xh_loss

        self.log_dict['G_loss'] = loss.item()
        self.log_dict['XH_loss'] = xh_loss.item()
        self.log_dict['X_loss'] = x_loss.item()
        self.log_dict['Y_loss'] = y_loss.item()
        self.log_dict['X_loss_last'] = x_loss_last.item()
        self.rx = rxs[-1]
        self.ry = rys[-1]
        loss.backward()

        self.optimizer.step()
