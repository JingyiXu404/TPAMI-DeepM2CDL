import math
import os
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import dirname, join

import scipy
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.special
from PIL import Image
"""
--------------------------------------------
Hongyi Zheng (github: https://github.com/natezhenghy)
07/Apr/2021
--------------------------------------------
Kai Zhang (github: https://github.com/cszn)
03/Mar/2019
--------------------------------------------
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR
--------------------------------------------
"""

##############
# path utils #
##############

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP', '.tif'
]


def is_img(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img_paths(dataroot: np.str0) -> List[str]:
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_img_paths_from_root(dataroot))
    return paths


def _get_img_paths_from_root(path: str) -> List[str]:
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images: List[str] = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_img(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def makedirs(paths: Union[str, List[str]]):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


###############
# image utils #
###############

def imread_uint_sr(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_Ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img_Ycrcb[:, :, 0]
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img
def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img


def imsave(img: np.ndarray, img_path: str):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def uint2single(img: np.ndarray) -> np.ndarray:
    return np.float32(img / 255.)


def uint2tensor3(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)
    return tensor


def tensor2uint(img: torch.Tensor) -> np.ndarray:
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def single2tensor3(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def save_d(d: np.ndarray, path: str = ''):
    def merge_images(image_batch: np.ndarray):
        """
            d: C_out, C_in, d_size, d_size
        """
        h, w = image_batch.shape[-2], image_batch.shape[-1]
        img = np.zeros((int(h * 8 + 7), int(w * 8 + 7)))
        for idx, im in enumerate(image_batch):
            i = idx % 8 * (h + 1)
            j = idx // 8 * (w + 1)

            img[j:j + h, i:i + w] = im
        img = cv2.resize(img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        return img

    d = np.where(d > np.quantile(d, 0.75), 0, d)
    d = np.where(d < np.quantile(d, 0.25), 0, d)

    im_merged = merge_images(d)
    im_merged = np.absolute(im_merged)
    plt.imsave(path,
               im_merged,
               cmap='Greys',
               vmin=im_merged.min(),
               vmax=im_merged.max())

def save_d_sort_l1(d: np.ndarray, path: str = ''):
    def merge_images(image_batch: np.ndarray):
        """
            d: C_out, C_in, d_size, d_size
        """
        h, w = image_batch.shape[-2], image_batch.shape[-1]
        img = np.zeros((int(h * 8 + 7), int(w * 8 + 7)))
        SUM = []
        for idx, im in enumerate(image_batch):
            im = np.absolute(im)
            sum = np.sum(im)
            SUM.append(sum)
        m_sorted = sorted(enumerate(SUM), key = lambda x:x[1])
        sorted_inds = [m[0] for m in m_sorted]
        print(sorted_inds)
        for num, im in enumerate(image_batch):
            idx = sorted_inds.index(num)
            i = idx % 8 * (h + 1)
            j = idx // 8 * (w + 1)
            img[j:j + h, i:i + w] = im
        img = cv2.resize(img,
                         dsize=(256, 256),
                         interpolation=cv2.INTER_NEAREST)
        return img

    d = np.where(d > np.quantile(d, 0.75), 0, d)
    d = np.where(d < np.quantile(d, 0.25), 0, d)

    im_merged = merge_images(d)
    im_merged = np.absolute(im_merged)
    plt.imsave(path,
               im_merged,
               cmap='Greys',
               vmin=im_merged.min(),
               vmax=im_merged.max())

######################
# augmentation utils #
######################
def augment_img(img: np.ndarray, mode: int = 0) -> np.ndarray:
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        raise ValueError


###########
# metrics #
###########
def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0):
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse: float = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,border: int = 0) -> float:
    if not img1.shape == img2.shape:
        img2 = img2[..., :img1.shape[-2], :img1.shape[-1]]
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims: List[float] = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')
    else:
        raise ValueError('Wrong input image dimensions.')
def calculate_niqe(img1: np.ndarray,border: int = 0):
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]

    return niqe(img1)

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    s: float = ssim_map.mean()
    return s

class Qabf():
    def __init__(self):
        self.L = 1
        self.Tg = 0.9994
        self.kg = -15
        self.Dg = 0.5
        self.Ta = 0.9879
        self.ka = -22
        self.Da = 0.8
        # Sobel Operator
        self.h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        self.h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        self.h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    # augment matrix 180
    def flip180(self,arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr

    # equal to Conv2 in matlab
    def convolution(self,k, data):
        k = self.flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        n, m = data.shape
        img_new = []
        for i in range(n - 2):
            line = []
            for j in range(m - 2):
                a = data[i:i + 3, j:j + 3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new)

    def getArray(self,img):
        SAx = self.convolution(self.h3, img)
        SAy = self.convolution(self.h1, img)
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if (SAx[i, j] == 0):
                    aA[i, j] = math.pi / 2
                else:
                    aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    def getQabf(self,aA,gA,aF,gF):
        n, m = aA.shape
        GAF = np.zeros((n,m))
        AAF = np.zeros((n,m))
        QgAF = np.zeros((n,m))
        QaAF = np.zeros((n,m))
        QAF = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if(gA[i,j]>gF[i,j]):
                    GAF[i,j] = gF[i,j]/gA[i,j]
                elif(gA[i,j]==gF[i,j]):
                    GAF[i, j] = gF[i, j]
                else:
                    GAF[i, j] = gA[i,j]/gF[i, j]
                AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

                QgAF[i,j] = self.Tg/(1+math.exp(self.kg*(GAF[i,j]-self.Dg)))
                QaAF[i,j] = self.Ta/(1+math.exp(self.ka*(AAF[i,j]-self.Da)))

                QAF[i,j] = QgAF[i,j]*QaAF[i,j]

        return QAF
def calculate_QABF(imgA,imgB,imgF):
    qabf = Qabf()
    if imgA.shape[2]>1:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        imgF = cv2.cvtColor(imgF, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    gA,aA = qabf.getArray(imgA)
    gB,aB = qabf.getArray(imgB)
    gF,aF = qabf.getArray(imgF)
    QAF = qabf.getQabf(aA, gA, aF, gF)
    QBF = qabf.getQabf(aB, gB, aF, gF)
    deno = np.sum(gA+gB)
    nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))
    QABF = nume/deno
    return QABF

###########
# MSSIM #
###########
from math import exp
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
###########
# MEFSSIM #
###########
def _mef_ssim(imgSeq, refImg, window, window_size):
    (_, imgSeq_channel, _, _) = imgSeq.size()
    (_, refImg_channel, _, _) = refImg.size()
    C2 = (0.03 * 255) ** 2
    sWindow = torch.ones((imgSeq_channel, 1, window_size, window_size)) / window_size ** 2
    mu_x = F.conv2d(imgSeq, sWindow, padding=window_size // 2, groups=imgSeq_channel)

    mfilter = torch.ones((imgSeq_channel, 1, window_size, window_size))
    x_hat = imgSeq - mu_x
    x_hat_norm = torch.sqrt(
        F.conv2d(torch.pow(x_hat, 2), mfilter, padding=window_size // 2, groups=imgSeq_channel)) + 0.001
    c_hat = torch.max(x_hat_norm, dim=1)[0]

    mfilter2 = torch.ones((1, 1, window_size, window_size))
    R = (torch.sqrt(F.conv2d(torch.pow(torch.sum(x_hat, 1, keepdim=True), 2), mfilter2, padding=window_size // 2,
                             groups=1)) + np.spacing(1) + np.spacing(1)) \
        / (torch.sum(x_hat_norm, 1, keepdim=True) + np.spacing(1))

    R[R > 1] = 1 - np.spacing(1)
    R[R < 0] = 0 + np.spacing(1)

    p = torch.tan(R * np.pi / 2)
    p[p > 10] = 10

    s = x_hat / x_hat_norm

    s_hat_one = torch.sum((torch.pow(x_hat_norm, p) + np.spacing(1)) * s, 1, keepdim=True) / torch.sum(
        (torch.pow(x_hat_norm, p) + np.spacing(1)), 1, keepdim=True)
    s_hat_two = s_hat_one / torch.sqrt(
        F.conv2d(torch.pow(s_hat_one, 2), mfilter2, padding=window_size // 2, groups=refImg_channel))

    x_hat_two = c_hat * s_hat_two

    mu_x_hat_two = F.conv2d(x_hat_two, window, padding=window_size // 2, groups=refImg_channel)
    mu_y = F.conv2d(refImg, window, padding=window_size // 2, groups=refImg_channel)

    mu_x_hat_two_sq = torch.pow(mu_x_hat_two, 2)
    mu_y_sq = torch.pow(mu_y, 2)
    mu_x_hat_two_mu_y = mu_x_hat_two * mu_y
    sigma_x_hat_two_sq = F.conv2d(x_hat_two * x_hat_two, window, padding=window_size // 2,
                                  groups=refImg_channel) - mu_x_hat_two_sq
    sigma_y_sq = F.conv2d(refImg * refImg, window, padding=window_size // 2, groups=refImg_channel) - mu_y_sq
    sigmaxy = F.conv2d(x_hat_two * refImg, window, padding=window_size // 2, groups=refImg_channel) - mu_x_hat_two_mu_y

    mef_ssim_map = (2 * sigmaxy + C2) / (sigma_x_hat_two_sq + sigma_y_sq + C2)

    return mef_ssim_map.mean()


class MEF_SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(MEF_SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img_seq, refImg):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img_seq.is_cuda:
                window = window.cuda(img_seq.get_device())
            window = window.type_as(img_seq)

        return _mef_ssim(img_seq, refImg, window, self.window_size)


def mef_ssim(img_seq, refImg, window_size=11):
    (_, channel, _, _) = refImg.size()
    window = create_window(window_size, channel)

    if img_seq.is_cuda:
        window = window.cuda(img_seq.get_device())
    window = window.type_as(img_seq)

    return _mef_ssim(img_seq, refImg, window, window_size)
def calculate_mefssim(A, B, F, window_size=11):
    print(F.shape)
    A = torch.tensor(A, dtype=torch.float)
    B = torch.tensor(B, dtype=torch.float)
    F = torch.tensor(F, dtype=torch.float)
    AB = torch.cat((A, B), dim=0)
    print(AB.shape)
    F = torch.unsqueeze(F, 0)
    AB = torch.unsqueeze(AB, 0)
    print(AB.shape,F.shape)
    return mef_ssim(AB,F, window_size=window_size)