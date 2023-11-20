import argparse
import faulthandler
import logging
import os
import os.path
import random
import time
from typing import Any, Dict, List

import numpy as np
import torch
from prettytable import PrettyTable
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.me_dataset import DatasetMultiExposure
from data.me_select_dataset import select_dataset
from models.MMIF_ME_model_s2_c2_r2 import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option

faulthandler.enable()
torch.autograd.set_detect_anomaly(True)


def main(json_path: str = 'options/MMIF_ME_train_s2_c2_r2.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=json_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])

    current_step = 0

    option.save(opt)

    # logger
    logger_name = 'train'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # seed
    seed = opt['train']['manual_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    # data
    opt_data_train: Dict[str, Any] = opt["data"]["train"]
    train_set: DatasetMultiExposure = select_dataset(opt_data_train, "train")
    train_loader: DataLoader[DatasetMultiExposure] = DataLoader(
        train_set,
        batch_size=opt_data_train['batch_size'],
        shuffle=True,
        num_workers=opt_data_train['num_workers'],
        drop_last=True,
        pin_memory=True)

    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetMultiExposure] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetMultiExposure]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=8,
                       drop_last=True,
                       pin_memory=True))

    # model
    #lam1=y_loss,lam2=0.1,lam3=xl_loss
    model = Model(opt,lam1=0.5,lam2=0.5,lam3=0.5)
    model.init()
    best_ssim=0 #104.05
    best_model=0
    # train
    start = time.time()
    GL = []
    XL = []
    XLL = []
    YL = []
    YLL = []
    XH = []
    XHL = []
    debug_num = 0
    for epoch in range(1,1000000):  # keep running
        for train_data in tqdm(train_loader):
            current_step += 1
            model.feed_data(train_data)

            model.train(i=epoch)

            model.update_learning_rate(current_step)

            if current_step % opt['train']['checkpoint_log'] == 0:
                GL,XL,XLL,YL,YLL,XH=model.log_train(current_step, epoch, logger,GL,XL,XLL,YL,YLL,XH,'debug/MMIF_ME_s2_c2_r2..png')

            if current_step % opt['train']['checkpoint_test'] == 0:
                avg_psnrs: Dict[str, List[float]] = {}
                avg_ssims: Dict[str, List[float]] = {}
                avg_qabfs: Dict[str, List[float]] = {}
                # avg_mefssims: Dict[str, List[float]] = {}
                tags: List[str] = []
                test_index = 0
                for test_loader in tqdm(test_loaders):
                    test_set: DatasetMultiExposure = test_loader.dataset
                    # avg_psnr = 0.
                    avg_ssim = 0.
                    avg_qabf = 0.
                    # avg_mefssim = 0.
                    for test_data in tqdm(test_loader):
                        test_index += 1
                        model.feed_data(test_data)
                        model.test()

                        qabf, ssim = model.cal_metrics()
                        # avg_psnr += psnr
                        avg_ssim += ssim
                        avg_qabf += qabf
                        # avg_mefssim += mefssim
                        if current_step % opt['train']['checkpoint_saveimage'] == 0:
                            model.save_visuals(test_set.tag)

                    # avg_psnr = round(avg_psnr / len(test_loader), 2)
                    avg_ssim = round(avg_ssim * 100 / len(test_loader), 2)
                    avg_qabf = round(avg_qabf / len(test_loader), 2)
                    # avg_mefssim = round(avg_mefssim * 100 / len(test_loader), 2)
                    
                    name = test_set.name_X
                    if name in avg_ssims:
                        # avg_psnrs[name].append(avg_psnr)
                        avg_ssims[name].append(avg_ssim)
                        avg_qabfs[name].append(avg_qabf)
                        # avg_mefssims[name].append(avg_mefssim)
                    else:
                        # avg_psnrs[name] = [avg_psnr]
                        avg_ssims[name] = [avg_ssim]
                        avg_qabfs[name] = [avg_qabf]
                        # avg_mefssims[name] = [avg_mefssim]
                    if test_set.tag not in tags:
                        tags.append(test_set.tag)
                    if avg_ssim> 103:
                        debug_num = debug_num + 1
                        model.save_debug(logger,debug_num)
                    if avg_ssim > best_ssim:
                        best_model = epoch
                        best_ssim = avg_ssim
                        model.save_best(logger)
                header = ['Dataset'] + tags
                t = PrettyTable(header)
                for key, value in avg_qabfs.items():
                    t.add_row(['qabf'] + value)
                for key, value in avg_ssims.items():
                    t.add_row(['ssim'] + value)
                logger.info(f"Test Qabf and SSIM:\n{t}")
                # t = PrettyTable(header)
                # for key, value in avg_mefssims.items():
                #     t.add_row([key] + value)
                # logger.info(f"Test MEFSSIM:\n{t}")

                logger.info(f"Time elapsed: {time.time() - start:.2f}")
                start = time.time()
                logger.info(f"best model:{best_model},best SSIM:{best_ssim}")
                model.save(logger)


if __name__ == '__main__':
    main()
