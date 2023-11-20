import random
from typing import Any, Dict, Union

import numpy as np
import torch
import utils.utils_image as util

from .mf_dataset_ir import DatasetIR
import cv2
import os
import random
from PIL import Image, ImageFilter
class MyGaussianBlur2(ImageFilter.Filter):
    name = "GaussianBlur2"

    def __init__(self, radius=2, w=None,h=None,width=None,height=None):
        self.radius = radius
        self.bounds = (w-width, h-height,w+width,h+height)

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius
        self.bounds=None

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)
def process_data(label,H=50,W=50,radius=3):
    image = Image.fromarray(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))
    (width, height) = image.size
    w = random.randrange(W, width - W)
    h = random.randrange(H, height - H)
    image_B = image.filter(MyGaussianBlur2(radius=radius, w=w, h=h, width=W, height=H))
    image_A = image.filter(MyGaussianBlur(radius=radius))
    sourceB = cv2.cvtColor(np.array(image_B), cv2.COLOR_RGB2BGR)
    sourceA = cv2.cvtColor(np.array(image_A), cv2.COLOR_RGB2BGR)
    sourceA[h - H:h + H, w - W:w + W] = label[h - H:h + H, w - W:w + W]
    return sourceA,sourceB
class DatasetMultiFocus(DatasetIR):
    def __init__(self, opt_dataset: Dict[str, Any]):
        super().__init__(opt_dataset)

        self.tag = str(self.down_scale)

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:

        if self.opt['phase'] == 'train':
            img_path = self.img_paths_X[index]
            img_H = util.imread_uint(img_path, self.n_channels)

            H, W = img_H.shape[:2]
            self.count += 1

            # crop
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            cut_H = random.randint(self.cut_HMIN, self.cut_HMAX)
            cut_W = random.randint(self.cut_WMIN, self.cut_WMAX)
            radius = random.randint(self.radius_MIN, self.radius_MAX)

            # augmentation
            mode=np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L,patch_Guide = process_data(patch_H,H=cut_H,W=cut_W,radius=radius)
            # print(patch_L.shape,patch_Guide.shape,patch_H.shape)

            # cv2.imwrite('A.png',patch_L)
            # cv2.imwrite('B.png',patch_Guide)
            # cv2.imwrite('GT.png', patch_H)

            # HWC to CHW, numpy(uint) to tensor
            img_Guide = util.uint2tensor3(patch_Guide)
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

            # scale_level: torch.FloatTensor = 1.0/torch.FloatTensor([self.down_scale[0]])
            scale_level: torch.FloatTensor = torch.FloatTensor([1.0/self.down_scale[0]])
            # print(scale_level)
            return {
                'y': img_L,
                'y_gt': img_H,
                'guide_gt': img_Guide,
                'down_scale': scale_level.unsqueeze(1).unsqueeze(1),
                'path': img_path,
                'path_l': img_path,
                'path_guide': img_path
            }

        else:
            img_path_L = self.img_paths_LX[index]
            img_path_guide = self.img_paths_Y[index]
            img_L = util.imread_uint(img_path_L, self.n_channels)
            img_Guide = util.imread_uint_sr(img_path_guide, self.n_channels)
            img_Guide = util.uint2single(img_Guide)
            img_L = util.uint2single(img_L)
            # scale_level: torch.FloatTensor = torch.FloatTensor([1.0 / self.down_scale])
            scale_level: torch.FloatTensor=torch.FloatTensor([1.0/self.down_scale])
            # print(scale_level)
            img_L,img_Guide = util.single2tensor3(img_L), util.single2tensor3(img_Guide)
            img_H = img_L
            return {
                'y': img_L,
                'y_gt': img_H,
                'guide_gt':img_Guide,
                'down_scale': scale_level.unsqueeze(1).unsqueeze(1),
                'path': img_path_L,
                'path_l': img_path_L,
                'path_guide':img_path_guide
            }
