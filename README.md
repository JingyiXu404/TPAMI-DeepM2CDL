# DeepM<sup>2</sup>CDL: Deep Multi-scale Multi-modal Convolutional Dictionary Learning Network
- This is the official repository of the paper "DeepM<sup>2</sup>CDL: Deep Multi-scale Multi-modal Convolutional Dictionary Learning Network" from **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**. [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/9774926, "Paper Link")

![framework](./imgs/framework1.png)
![framework](./imgs/framework2.png)


## 1. Environment
- Python >= 3.5
- PyTorch == 1.7.1 is recommended
- opencv-python = =3.4.9.31
- tqdm
- scikit-image == 0.15.0
- scipy == 1.3.1 
- Matlab

## 2. Training and testing dataset
- ***For flash guided non-flash image denoising task***, we randomly select image pairs from  Aksoy dataset for training (400 image pairs) and testing (12 image pairs). 
- ***For RGB guided depth image super-resolution task***, the training data is from the DPDN dataset and the testing datasets are from the Middlebury dataset and the Sintel dataset.  
- ***For multi-focus image fusion task***, we generate the training data from DIV2K dataset and the testing image pairs for multi-focus image fusion are from the Lytro Mutli-focus image dataset. 
- ***For multi-exposure image fusion task***, the training data are from the SICE dataset. The testing images are from SICE dataset, MEFB dataset and  PQA-MEF dataset.

All the training and testing images for different MIP tasks used in this paper can be downloaded from the [[Google Drive Link]](https://drive.google.com/drive/folders/1Dpjl7KPrDtrbstNjgxzwYjhvlu4OWlrb?usp=sharing)


## 3. Test
### 🛠️  Clone this repository:
```
    git clone https://github.com/JingyiXu404/TPAMI-DeepM2CDL.git
```
### 🛠️  Download pretrained models:
```
    https://drive.google.com/drive/folders/1Sef-rFosbzu40h9NH3J1wDyd-w4clfAj?usp=sharing
```
### 💓  For flash guided non-flash image denoising task
**1. Prepare dataset**: If you do not use same datasets as us, place the test images in `Flash_Guide_Nonflash_Denoise/code/data/denoise/`.

```
    denoise
    └── test_flash
        └── flash
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
    └── test_nonflash
        └── nonflash
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
   ```

**2. Setup configurations**: In `Flash_Guide_Nonflash_Denoise/code/options/MMIR_test_denoising_new.json`.

```
    "root": "debug/N25"
    "pretrained_netG": "../Results_models/N25/"
    "sigma": [25], // noise level [25,50,75]}
```

**3. Run**: 

```
   cd Flash_Guide_Nonflash_Denoise/code/
   python MMIR_test_dcdicl.py
```

### 🐍 For RGB guided depth image super-resolution task
**1. Prepare dataset**: If you do not use same datasets as us, place the test images in `RGB_Guide_Depth_Super-resolution/code/data/depth_sr/`.

```
    denoise
    └── test_depth
        └── depth
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
    └── test_lr
        └── depth_lr
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
    └── test_rgb
        └── rgb
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
   ```

**2. Run**: 

```
   cd RGB_Guide_Depth_Super-resolution/code/
   python MMIR_test_dcdicl_sr.py
```
### 🧪 For multi-focus image fusion task
**1. Prepare dataset**: If you do not use same datasets as us, place the test images in `Multi-Focus_Fusion/code/data/multi-focus/`.

```
    denoise
    └── test_A
        └── source_A
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
    └── test_B
        └── source_B
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
   ```

**2. Run**: 

```
   cd Multi-Focus_Fusion/code/
   python MMIF_test_mf.py
```

## 4. Citation
If you find our work useful in your research or publication, please cite our work:
```
release soon
```

## 5. Contact
If you have any question about our work or code, please email `jingyixu@buaa.edu.cn` .
