import random
from typing import Any, Dict, Union

import numpy as np
import torch
import utils.utils_image as util

from .sr_dataset_ir import DatasetIR
import cv2

class DatasetSuperResolution(DatasetIR):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__(opt_dataset)

        self.tag = str(self.down_scale)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        img_path = self.img_paths_X[index]
        img_path_L = self.img_paths_LX[index]
        img_path_guide = self.img_paths_Y[index]

        img_H = util.imread_uint(img_path, self.n_channels)
        img_L = util.imread_uint(img_path_L, self.n_channels)

        img_Guide = util.imread_uint_sr(img_path_guide, self.n_channels)
        # # RGB2Ycrcb choose Y
        # img_H_Ycrcb = cv2.cvtColor(img_H, cv2.COLOR_RGB2YCrCb)
        # img_L_Ycrcb = cv2.cvtColor(img_L, cv2.COLOR_RGB2YCrCb)
        # img_Guide_Ycrcb = cv2.cvtColor(img_Guide, cv2.COLOR_RGB2YCrCb)
        #
        # img_H = img_H_Ycrcb[:, :, 0]
        # img_L = img_L_Ycrcb[:, :, 0]
        # img_Guide = img_Guide_Ycrcb[:, :, 0]
        #
        # img_H = np.expand_dims(img_H, axis=2)
        # img_L = np.expand_dims(img_L, axis=2)
        # img_Guide = np.expand_dims(img_Guide, axis=2)

        H, W = img_H.shape[:2]

        if self.opt['phase'] == 'train':

            self.count += 1

            # crop
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_Guide = img_Guide[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation
            mode=np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)
            patch_Guide = util.augment_img(patch_Guide, mode=mode)

            # HWC to CHW, numpy(uint) to tensor
            img_Guide = util.uint2tensor3(patch_Guide)
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

            # scale_level: torch.FloatTensor = 1.0/torch.FloatTensor([self.down_scale[0]])
            scale_level: torch.FloatTensor = torch.FloatTensor([1.0/self.down_scale[0]])
            # print(scale_level)

        else:
            img_Guide = util.uint2single(img_Guide)
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)
            # scale_level: torch.FloatTensor = torch.FloatTensor([1.0 / self.down_scale])
            scale_level: torch.FloatTensor=torch.FloatTensor([1.0/self.down_scale])
            # print(scale_level)
            img_H, img_L,img_Guide = util.single2tensor3(img_H), util.single2tensor3(img_L), util.single2tensor3(img_Guide)

        return {
            'y': img_L,
            'y_gt': img_H,
            'guide_gt':img_Guide,
            'down_scale': scale_level.unsqueeze(1).unsqueeze(1),
            'path': img_path,
            'path_l': img_path_L,
            'path_guide':img_path_guide
        }
