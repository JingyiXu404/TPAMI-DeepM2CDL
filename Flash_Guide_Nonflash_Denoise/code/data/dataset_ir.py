import os
from typing import Any, Dict

import torch.utils.data as data
import utils.utils_image as util


class DatasetIR(data.Dataset):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__()

        self.opt = opt_dataset

        if self.opt['phase'] == 'train':
            self.patch_size = self.opt['H_size']
        self.n_channels = opt_dataset['n_channels']
        self.sigma = opt_dataset['sigma']

        self.name_X: str = os.path.basename(opt_dataset['dataroot_HX'])
        self.img_paths_X = util.get_img_paths(opt_dataset['dataroot_HX'])
        self.name_Y: str = os.path.basename(opt_dataset['dataroot_HY'])
        self.img_paths_Y = util.get_img_paths(opt_dataset['dataroot_HY'])
        # print('name_X',self.name_X)
        # print('img_paths_X',len(self.img_paths_X))
        # print('name_Y',self.name_Y)
        # print('img_paths_Y',len(self.img_paths_Y))
        self.count = 0
        self.tag: str = ""

    def __len__(self):
        return len(self.img_paths_X)
