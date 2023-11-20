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
            self.cut_HMIN = self.opt['cut_HMIN']
            self.cut_WMIN = self.opt['cut_WMIN']
            self.cut_HMAX = self.opt['cut_HMAX']
            self.cut_WMAX = self.opt['cut_WMAX']
            self.radius_MIN = self.opt['radius_MIN']
            self.radius_MAX = self.opt['radius_MAX']
            self.name_X: str = os.path.basename(opt_dataset['dataroot_HX'])
            self.img_paths_X = util.get_img_paths(opt_dataset['dataroot_HX'])
            self.name_LX = self.name_X
            self.name_Y = self.name_X
            self.img_paths_LX = self.img_paths_X
            self.img_paths_Y = self.img_paths_X
        else:
            self.name_LX: str = os.path.basename(opt_dataset['dataroot_LX'])
            self.img_paths_LX = util.get_img_paths(opt_dataset['dataroot_LX'])
            self.name_Y: str = os.path.basename(opt_dataset['dataroot_HY'])
            self.img_paths_Y = util.get_img_paths(opt_dataset['dataroot_HY'])
            self.name_X: str =self.name_LX
            self.img_paths_X = self.img_paths_LX
        self.n_channels = opt_dataset['n_channels']
        self.down_scale = opt_dataset['down_scale']


        self.count = 0
        self.tag: str = ""

    def __len__(self):
        return len(self.img_paths_X)
