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


def main(config_path: str = 'options/MMIF_test_me.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=config_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])

    option.save(opt)

    # logger
    logger_name = 'test'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # data
    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetMultiFocus] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetMultiFocus]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=8,
                       drop_last=False,
                       pin_memory=True))

    # model
    model = Model(opt,lam1=0.5,lam2=0.1,lam3=0.5)
    model.init()

    # test
    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    avg_qabfs: Dict[str, List[float]] = {}
    tags = []
    for test_loader in test_loaders:
        test_set: DatasetMultiFocus = test_loader.dataset
        # avg_psnr = 0.
        avg_ssim = 0.
        avg_qabf = 0.
        for test_data in test_loader:
            model.feed_data(test_data)
            model.test()

            qabf, ssim= model.cal_metrics()
            # avg_psnr += psnr
            avg_ssim += ssim
            avg_qabf += qabf
            model.save_visuals(test_set.tag)

        # avg_psnr = round(avg_psnr / len(test_loader), 2)
        avg_ssim = round(avg_ssim*100 / len(test_loader), 2)
        avg_qabf = round(avg_qabf / len(test_loader), 2)

        name = test_set.name_X

        if name in avg_ssims:
            # avg_psnrs[name].append(avg_psnr)
            avg_ssims[name].append(avg_ssim)
            avg_qabfs[name].append(avg_qabf)
        else:
            # avg_psnrs[name] = [avg_psnr]
            avg_ssims[name] = [avg_ssim]
            avg_qabfs[name] = [avg_qabf]

        tags.append(test_set.tag)

    header = ['Dataset'] + list(set(tags))

    t = PrettyTable(header)
    for key, value in avg_qabfs.items():
        t.add_row(['qabf'] + value)
    for key, value in avg_ssims.items():
        t.add_row(['ssim'] + value)
    logger.info(f"Test SSIM:\n{t}")



main()
