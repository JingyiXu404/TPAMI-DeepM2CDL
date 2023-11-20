import copy
from logging import Logger

from models.MMIF_network import DCDicL
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
from models.MMIF_ME_select_network import select_network
from utils import utils_image as util
import freeze
import matplotlib
import pdb
matplotlib.use('agg')
class Model:
    def __init__(self, opt: Dict[str, Any],lam1:float,lam2:float,lam3:float):
        self.opt = opt
        self.opt_train = self.opt['train']
        self.opt_test = self.opt['test']
        self.c_iter = opt['netG']['c_iter']
        self.x_iter = opt['netG']['x_iter']
        self.y_iter = opt['netG']['y_iter']
        self.rnn_iter = opt['netG']['rnn_iter']
        self.channels = opt['netG']['nc_x'][0]
        self.save_dir: str = opt['path']['models']
        self.debug_num = 0
        if self.opt_test['is_test']:
            self.debug_num = self.opt_test['debug_num']

        self.is_train = opt['is_train']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.type = opt['netG']['type']
        self.save_feature = self.opt_test['save_feature']
        self.save_dictionary = self.opt_test['save_dictionary']
        self.save_image = self.opt_test['save_image']
        self.is_debug = self.opt_test['is_debug']
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

        for j in range(self.c_iter):
                network.headx.head_list[j].load_state_dict(torch.load(load_path + 'headx_c' + str(j) + '.pth'), strict=True)
                network.heady.head_list[j].load_state_dict(torch.load(load_path + 'heady_c' + str(j) + '.pth'), strict=True)
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                network.FB_list[i].FB_list[j].update_feature_x.load_state_dict(torch.load(load_path + 'UFBx_r' + str(i) +'_c'+str(j) + '.pth'), strict=True)
                network.FB_list[i].FB_list[j].update_feature_y.load_state_dict(torch.load(load_path + 'UFBy_r' + str(i) +'_c'+str(j) + '.pth'), strict=True)
                network.FB_list[i].FB_list[j].update_dictionary_x.load_state_dict(torch.load(load_path + 'UDBx_r' + str(i) +'_c'+str(j) + '.pth'), strict=True)
                network.FB_list[i].FB_list[j].update_dictionary_y.load_state_dict(torch.load(load_path + 'UDBy_r' + str(i) +'_c'+str(j) + '.pth'), strict=True)
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
            self.save_network(net.headx.head_list[j], 'headx_c' + str(j))
            self.save_network(net.heady.head_list[j], 'heady_c' + str(j))
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                self.save_network(net.FB_list[i].FB_list[j].update_feature_x, 'UFBx_r' + str(i) +'_c'+str(j))
                self.save_network(net.FB_list[i].FB_list[j].update_feature_y, 'UFBy_r' + str(i) +'_c'+str(j))
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_x, 'UDBx_r' + str(i) +'_c'+str(j))
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_y, 'UDBy_r' + str(i) +'_c'+str(j))
                self.save_network(net.body_listx[i].scale_list[j].body.net_x, 'ax_r' + str(i) + '_c' + str(j))
                self.save_network(net.body_listx[i].scale_list[j].body.net_d, 'dx_r' + str(i) + '_c' + str(j))
                self.save_network(net.body_listy[i].scale_list[j].body.net_x, 'ay_r' + str(i) + '_c' + str(j))
                self.save_network(net.body_listy[i].scale_list[j].body.net_d, 'dy_r' + str(i) + '_c' + str(j))
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
            self.save_network(net.headx.head_list[j], 'headx_c' + str(j), best=True)
            self.save_network(net.heady.head_list[j], 'heady_c' + str(j), best=True)
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                self.save_network(net.FB_list[i].FB_list[j].update_feature_x, 'UFBx_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.FB_list[i].FB_list[j].update_feature_y, 'UFBy_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_x, 'UDBx_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_y, 'UDBy_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.body_listx[i].scale_list[j].body.net_x, 'ax_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.body_listx[i].scale_list[j].body.net_d, 'dx_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.body_listy[i].scale_list[j].body.net_x, 'ay_r' + str(i) + '_c' + str(j), best=True)
                self.save_network(net.body_listy[i].scale_list[j].body.net_d, 'dy_r' + str(i) + '_c' + str(j), best=True)
                for m in range(self.x_iter):
                    self.save_network(net.body_listx[i].scale_list[j].hypa_list[m], 'hypax_r' + str(i) + '_c' + str(j) + '_i' + str(m), best=True)
                for m in range(self.y_iter):
                    self.save_network(net.body_listy[i].scale_list[j].hypa_list[m], 'hypay_r' + str(i) + '_c' + str(j) + '_i' + str(m), best=True)
    def save_debug(self, logger: Logger,debug_num):
        logger.info('Saving the model.')
        net = self.net
        if isinstance(net, nn.DataParallel):
            net = net.module
        for j in range(self.c_iter):
            self.save_network(net.headx.head_list[j], 'headx_c' + str(j), debug=True, debug_num=debug_num)
            self.save_network(net.heady.head_list[j], 'heady_c' + str(j), debug=True, debug_num=debug_num)
        for i in range(self.rnn_iter):
            for j in range(self.c_iter):
                self.save_network(net.FB_list[i].FB_list[j].update_feature_x, 'UFBx_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.FB_list[i].FB_list[j].update_feature_y, 'UFBy_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_x, 'UDBx_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.FB_list[i].FB_list[j].update_dictionary_y, 'UDBy_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.body_listx[i].scale_list[j].body.net_x, 'ax_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.body_listx[i].scale_list[j].body.net_d, 'dx_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.body_listy[i].scale_list[j].body.net_x, 'ay_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                self.save_network(net.body_listy[i].scale_list[j].body.net_d, 'dy_r' + str(i) + '_c' + str(j), debug=True, debug_num=debug_num)
                for m in range(self.x_iter):
                    self.save_network(net.body_listx[i].scale_list[j].hypa_list[m], 'hypax_r' + str(i) + '_c' + str(j) + '_i' + str(m), debug=True, debug_num=debug_num)
                for m in range(self.y_iter):
                    self.save_network(net.body_listy[i].scale_list[j].hypa_list[m], 'hypay_r' + str(i) + '_c' + str(j) + '_i' + str(m), debug=True, debug_num=debug_num)
    def save_network(self, network, network_label,best=False,debug=False,debug_num=0):
        filename = '{}.pth'.format(network_label)
        if best:
            os.makedirs(self.save_dir + '/best/', exist_ok=True)
            save_path = os.path.join(self.save_dir+'/best/', filename)
        elif debug:
            os.makedirs(self.save_dir + '/debug/' + str(debug_num) +'/', exist_ok=True)
            save_path = os.path.join(self.save_dir + '/debug/' + str(debug_num) +'/', filename)
        else:
            os.makedirs(self.save_dir + '/latest/', exist_ok=True)
            save_path = os.path.join(self.save_dir+'/latest/', filename)
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
        self.sigma = data['down_scale'].to(self.device)
        self.sigmay = data['down_scale'].to(self.device)
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
    def cal_multi_loss2(self, preds: List[torch.Tensor],gt: torch.Tensor,lam:float) -> torch.Tensor:
        losses = None
        for i, pred in enumerate(preds):
            loss = self.lossfn(pred, gt)
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses

    def log_train(self, current_step: int, epoch: int, logger: Logger,GL:list,XL:list,XLL:list,YL:list,YLL:list,XH:list,name:str):
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
            elif k == 'Y_loss_last':
                YLL.append(v)
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
        plt.plot(YLL)
        plt.ylabel('YLL')
        plt.subplot(3, 2, 6)
        plt.plot(XH)
        plt.ylabel('XH')

        plt.savefig(name)
        plt.close('all')
        return GL,XL,XLL,YL,YLL,XH
    def test(self):
        self.net.eval()
        # for name, child in self.net.module.body_listx[0].scale_list.named_children():
        #     print(name,child)
        with torch.no_grad():
            y = self.y
            guide = self.guide_gt
            # h, w = y.size()[-2:]
            # top = slice(0, h // 8 * 8)
            # left = slice(0, (w // 8 * 8))
            # y = y[..., top, left]
            # guide = guide[..., top, left]

            self.rh,self.rx,self.ry,self.DX,self.DY,self.AX,self.AY,self.BX,self.EY,self.FA,self.GA = self.net(y,guide, self.sigma,self.sigmay)

        self.prepare_visuals()

        self.net.train()

    def prepare_visuals(self):
        """ prepare visual for first sample in batch """
        self.out_dict = {}
        self.out_dict['y'] = util.tensor2uint(self.y[0].detach().float().cpu())
        self.out_dict['guide'] = util.tensor2uint(self.guide_gt[0].detach().float().cpu())
        self.out_dict['path'] = self.path[0]
        self.out_dict['y_gt'] = util.tensor2uint(self.y_gt[0].detach().float().cpu())
        if self.is_debug:
            self.out_dict['rh_r' + str(self.rnn_iter-1)] = util.tensor2uint(self.rh[self.rnn_iter-1].detach().float().cpu())
        else:

            for i in range(self.rnn_iter):
                if self.save_image:
                    self.out_dict['rx_r' + str(i)] = util.tensor2uint(self.rx[i].detach().float().cpu())
                    self.out_dict['ry_r' + str(i)] = util.tensor2uint(self.ry[i].detach().float().cpu())
                    self.out_dict['rh_r' + str(i)] = util.tensor2uint(self.rh[i].detach().float().cpu())
                for j in range(self.c_iter):
                    if self.save_dictionary:
                        self.out_dict['dx_r' + str(i) + '_c' + str(j)] = self.DX[i][j][0].detach().float().cpu()
                        self.out_dict['dy_r' + str(i) + '_c' + str(j)] = self.DY[i][j][0].detach().float().cpu()
                        self.out_dict['bx_r' + str(i) + '_c' + str(j)] = self.BX[i][j][0].detach().float().cpu()
                        self.out_dict['ey_r' + str(i) + '_c' + str(j)] = self.EY[i][j][0].detach().float().cpu()
                    if self.save_feature:
                        for k in range(self.channels):
                            self.out_dict['ax_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.AX[i][j][0][k:k+1, ...].detach().float().cpu())
                            self.out_dict['ay_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.AY[i][j][0][k:k+1, ...].detach().float().cpu())
                            self.out_dict['fa_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.FA[i][j][0][k:k+1, ...].detach().float().cpu())
                            self.out_dict['ga_r' + str(i) + '_c' + str(j) + '_' + str(k)] = util.tensor2uint(self.GA[i][j][0][k:k + 1, ...].detach().float().cpu())

    def cal_metrics(self):
        # self.metrics['psnr'] = util.calculate_psnr(self.out_dict['rxh'],self.out_dict['y_gt'])
        # self.metrics['ssim'] = util.calculate_ssim(self.out_dict['rxh'],self.out_dict['y_gt'])
        # self.metrics['psnr'] = util.calculate_psnr(self.out_dict['rh_r1'], self.out_dict['y_gt'])
        self.metrics['Qabf']=util.calculate_QABF(self.out_dict['y'],self.out_dict['guide'],self.out_dict['rh_r1'])
        self.metrics['ssim'] = util.calculate_ssim(self.out_dict['rh_r1'], self.out_dict['y']) + util.calculate_ssim(self.out_dict['rh_r1'], self.out_dict['guide'])
        return self.metrics['Qabf'],self.metrics['ssim']
    def visual_features(self,img):
        result = np.zeros(img.shape, dtype=np.float32)
        cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        result = np.uint8(result * 255.0)
        return result
    def save_visuals(self, tag: str):
        gt_img = self.out_dict['y_gt']
        y_img = self.out_dict['y']
        guide_img = self.out_dict['guide']
        rh_img = []
        rx_img = []
        ry_img = []
        ax_img = []
        ay_img = []
        fa_img = []
        ga_img = []
        dx_img = []
        dy_img = []
        bx_img = []
        ey_img = []
        if self.is_debug:
            rh_img = self.out_dict['rh_r' + str(self.rnn_iter-1)]
        else:

            for i in range(self.rnn_iter):
                if self.save_image:
                    rx_img.append(self.out_dict['rx_r' + str(i)])
                    ry_img.append(self.out_dict['ry_r' + str(i)])
                    rh_img.append(self.out_dict['rh_r' + str(i)])
                dx_img_sub = []
                dy_img_sub = []
                bx_img_sub = []
                ey_img_sub = []
                ax_img_sub = []
                ay_img_sub = []
                fa_img_sub = []
                ga_img_sub = []
                for j in range(self.c_iter):
                    if self.save_dictionary:
                        dx_img_sub.append(self.out_dict['dx_r' + str(i) + '_c' + str(j)])
                        dy_img_sub.append(self.out_dict['dy_r' + str(i) + '_c' + str(j)])
                        bx_img_sub.append(self.out_dict['bx_r' + str(i) + '_c' + str(j)])
                        ey_img_sub.append(self.out_dict['ey_r' + str(i) + '_c' + str(j)])
                    if self.save_feature:
                        ax_img_sub_sub = []
                        ay_img_sub_sub = []
                        fa_img_sub_sub = []
                        ga_img_sub_sub = []
                        for k in range(self.channels):
                            ax_img_sub_sub.append(self.out_dict['ax_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                            ay_img_sub_sub.append(self.out_dict['ay_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                            fa_img_sub_sub.append(self.out_dict['fa_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                            ga_img_sub_sub.append(self.out_dict['ga_r' + str(i) + '_c' + str(j) + '_' + str(k)])
                        ax_img_sub.append(ax_img_sub_sub)
                        ay_img_sub.append(ay_img_sub_sub)
                        fa_img_sub.append(fa_img_sub_sub)
                        ga_img_sub.append(ga_img_sub_sub)
                dx_img.append(dx_img_sub)
                dy_img.append(dy_img_sub)
                bx_img.append(bx_img_sub)
                ey_img.append(ey_img_sub)
                ax_img.append(ax_img_sub)
                ay_img.append(ay_img_sub)
                fa_img.append(fa_img_sub)
                ga_img.append(ga_img_sub)

        path = self.out_dict['path']
        img_name = os.path.splitext(os.path.basename(path))[0]


        if self.is_debug:
            if self.debug_num == 0:
                util.imsave(rh_img, os.path.join(self.opt['path']['images'], img_name + '_h_r' + str(self.rnn_iter) + '.png'))
            else:
                img_dir = os.path.join(self.opt['path']['images'], 'debug', self.debug_num)
                os.makedirs(img_dir, exist_ok=True)
                util.imsave(rh_img, os.path.join(img_dir, img_name + '_h_r' + str(self.rnn_iter) + '.png'))
        else:
            img_dir = os.path.join(self.opt['path']['images'], img_name)
            os.makedirs(img_dir, exist_ok=True)

            old_img_path = os.path.join(img_dir, f"{img_name:s}_{tag}_*_*.png")
            old_img = glob(old_img_path)
            for img in old_img: os.remove(img)

            img_path = os.path.join(
                img_dir,
                f"{img_name}_{tag}_{self.metrics['Qabf']}.png"
            )
            for i in range(self.rnn_iter):
                if self.save_image:
                    util.imsave(rx_img[i], img_path.replace('.png', '_x_r' + str(i + 1) + '.png'))
                    util.imsave(ry_img[i], img_path.replace('.png', '_y_r' + str(i + 1) + '.png'))
                    util.imsave(rh_img[i], img_path.replace('.png', '_h_r' + str(i + 1) + '.png'))
                for j in range(self.c_iter):
                    if self.save_feature:
                        os.makedirs(os.path.join(img_dir, 'ax_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
                        os.makedirs(os.path.join(img_dir, 'ay_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
                        os.makedirs(os.path.join(img_dir, 'fa_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
                        os.makedirs(os.path.join(img_dir, 'ga_r' + str(i + 1) + '_c' + str(j + 1)), exist_ok=True)
                        for k in range(self.channels):
                            util.imsave(self.visual_features(ax_img[i][j][k]),os.path.join(img_dir, 'ax_r' + str(i + 1) + '_c' + str(j + 1), str(k + 1) + '.png'))
                            util.imsave(self.visual_features(ay_img[i][j][k]),os.path.join(img_dir, 'ay_r' + str(i + 1) + '_c' + str(j + 1), str(k + 1) + '.png'))
                            util.imsave(self.visual_features(fa_img[i][j][k]),os.path.join(img_dir, 'fa_r' + str(i + 1) + '_c' + str(j + 1), str(k + 1) + '.png'))
                            util.imsave(self.visual_features(ga_img[i][j][k]),os.path.join(img_dir, 'ga_r' + str(i + 1) + '_c' + str(j + 1), str(k + 1) + '.png'))
            util.imsave(y_img, img_path.replace('.png', '_noise.png'))
            util.imsave(gt_img, img_path.replace('.png', '_gt.png'))
            util.imsave(guide_img, img_path.replace('.png', '_guide.png'))
            if self.opt['test']['visualize']:
                for i in range(self.rnn_iter):
                    for j in range(self.c_iter):
                        if self.save_dictionary:
                            util.save_d(dx_img[i][j].mean(0).numpy(), img_path.replace('.png', '_dx_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))
                            util.save_d(dy_img[i][j].mean(0).numpy(), img_path.replace('.png', '_dy_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))
                            util.save_d(bx_img[i][j].mean(0).numpy(),img_path.replace('.png', '_bx_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))
                            util.save_d(ey_img[i][j].mean(0).numpy(), img_path.replace('.png', '_ey_r' + str(i + 1) + '_c' + str(j + 1) + '.png'))

    def get_freeze_head(self):
        import freeze
        freeze.freeze_by_names(self.net.module.headx.head_list, ('0'))
        freeze.freeze_by_names(self.net.module.headx.head_list, ('1'))
        freeze.freeze_by_names(self.net.module.heady.head_list, ('0'))
        freeze.freeze_by_names(self.net.module.heady.head_list, ('1'))
    def get_unfreeze_head(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.headx.head_list, ('0'))
        freeze.unfreeze_by_names(self.net.module.headx.head_list, ('1'))
        freeze.unfreeze_by_names(self.net.module.heady.head_list, ('0'))
        freeze.unfreeze_by_names(self.net.module.heady.head_list, ('1'))
    def get_freeze_r1_c1(self):
        import freeze
        freeze.freeze_by_names(self.net.module.body_listx[0].scale_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listy[0].scale_list, ('0'))
    def get_unfreeze_r1_c1(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.body_listx[0].scale_list, ('0'))
        freeze.unfreeze_by_names(self.net.module.body_listy[0].scale_list, ('0'))
    def get_freeze_r1_c2(self):
        import freeze
        freeze.freeze_by_names(self.net.module.body_listx[0].scale_list, ('1'))
        freeze.freeze_by_names(self.net.module.body_listy[0].scale_list, ('1'))
    def get_unfreeze_r1_c2(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.body_listx[0].scale_list, ('1'))
        freeze.unfreeze_by_names(self.net.module.body_listy[0].scale_list, ('1'))
    def get_freeze_r2_c1(self):
        import freeze
        freeze.freeze_by_names(self.net.module.body_listx[1].scale_list, ('0'))
        freeze.freeze_by_names(self.net.module.body_listy[1].scale_list, ('0'))
    def get_unfreeze_r2_c1(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.body_listx[1].scale_list, ('0'))
        freeze.unfreeze_by_names(self.net.module.body_listy[1].scale_list, ('0'))
    def get_freeze_r2_c2(self):
        import freeze
        freeze.freeze_by_names(self.net.module.body_listx[1].scale_list, ('1'))
        freeze.freeze_by_names(self.net.module.body_listy[1].scale_list, ('1'))
    def get_unfreeze_r2_c2(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.body_listx[1].scale_list, ('1'))
        freeze.unfreeze_by_names(self.net.module.body_listy[1].scale_list, ('1'))
    def get_freeze_FB_r1_c1(self):
        import freeze
        freeze.freeze_by_names(self.net.module.FB_list[0].FB_list, ('0'))
    def get_unfreeze_FB_r1_c1(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.FB_list[0].FB_list, ('0'))
    def get_freeze_FB_r1_c2(self):
        import freeze
        freeze.freeze_by_names(self.net.module.FB_list[0].FB_list, ('1'))
    def get_unfreeze_FB_r1_c2(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.FB_list[0].FB_list, ('1'))
    def get_freeze_FB_r2_c1(self):
        import freeze
        freeze.freeze_by_names(self.net.module.FB_list[1].FB_list, ('0'))
    def get_unfreeze_FB_r2_c1(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.FB_list[1].FB_list, ('0'))
    def get_freeze_FB_r2_c2(self):
        import freeze
        freeze.freeze_by_names(self.net.module.FB_list[1].FB_list, ('1'))
    def get_unfreeze_FB_r2_c2(self):
        import freeze
        freeze.unfreeze_by_names(self.net.module.FB_list[1].FB_list, ('1'))

    def train(self,i):
        self.optimizer.zero_grad()
        rxhs,rxs, rys = self.net(self.y, self.guide_gt,self.sigma,self.sigmay)
        # loss
        xh_loss = self.lossfn(rxhs[-1], self.y_gt)
        # x_loss=self.cal_multi_loss2(rxs, self.y,lam=self.lam2)
        # y_loss=self.cal_multi_loss2(rys, self.guide_gt,lam=self.lam2)
        x_loss = self.cal_multi_loss2([rxs[-1]], self.y_gt,lam=self.lam2)
        y_loss = self.cal_multi_loss2([rys[-1]], self.y_gt,lam=self.lam2)
        x_loss_last = self.lossfn(rxs[-1], self.y)
        y_loss_last = self.lossfn(rys[-1], self.guide_gt)
        loss = xh_loss + 0.5 * x_loss + 0.5 * y_loss


        self.log_dict['G_loss'] = loss.item()
        self.log_dict['XH_loss'] = xh_loss.item()
        self.log_dict['X_loss'] = x_loss.item()
        self.log_dict['Y_loss'] = y_loss.item()
        self.log_dict['X_loss_last'] = x_loss_last.item()
        self.log_dict['Y_loss_last'] = y_loss_last.item()
        self.rxh = rxhs[-1]
        self.rx = rxs[-1]
        self.ry = rys[-1]
        loss.backward()

        self.optimizer.step()
