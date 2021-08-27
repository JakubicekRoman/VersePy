import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as im
# import nibabel as nib

import SimpleITK as sitk
import medpy as medpy
import torch
# import pandas as pd

import torch
from torch.utils import data
import glob
import os
import cv2


class DataLoader():
    def __init__(self, path_data="C:/Data/Verse2020"):      
        self.pat_list = glob.glob(path_data + "/*/")
        self.n = len(self.pat_list)
        self.data_list = [""]*self.n
        self.mask_list = [""]*self.n
        # print(self.pat_list[0])
        # self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
        # print(' '.join(self.name[0]))
        
        for i in range(self.n):
            # self.data_list[0] = self.name[0];
            name = glob.glob(self.pat_list[i] + "/*nii.gz")
            self.data_list[i] = "".join(name[0])
            self.mask_list[i] = "".join(name[1])
            
    def __getitem__(self, index):
        img, H = medpy.io.load(self.data_list[index])
        # img = img[:,:,100]
        # img = cv2.resize(img, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
        img = torch.tensor(((img)+1024)/4096)

        mask, H = medpy.io.load(self.mask_list[index])
        # mask = mask[:,:,100]
        # mask = cv2.resize(mask, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
        mask = torch.tensor(mask.astype(np.bool_))
        
        return img, mask

loader = DataLoader(path_data="C:/Data/Verse2020")


###### testovani funkcnosti
img, mask = loader[0]

plt.figure()
plt.imshow(img[:,:,105],cmap='gray')
plt.figure()
plt.imshow(mask[:,:,105],cmap='gray')


####### blbosti na testovani

# path_data = r"C:\Data\Verse2020\verse122\verse122.nii.gz"
# img, H = medpy.io.load(path_data)
# img = img[:,:,100]
# img = cv2.resize(img, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
# img = torch.tensor(((img)+1024)/4096)

# img1 = img.reshape(1,168,168)
# img = np.res(img,(64,64))
# img.resize((28,28))
# img = im.resize((28,28))

# plt.figure()
# plt.imshow(img,cmap='gray')

# path_mask = r"C:\Data\Verse2020\verse122\verse122_seg.nii.gz"
# mask, Hm = medpy.io.load(path_mask)

# plt.figure()
# plt.imshow(img[:,:,100],cmap='gray')
# plt.show()
# plt.imshow(mask[:,:,100],cmap='gray')
# plt.show()

# fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
# ax1.imshow(montage(img), cmap ='bone')