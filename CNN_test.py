import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
# import matplotlib.image as im
# import nibabel as nib

import SimpleITK as sitk
# import medpy as medpy
# import pandas as pd
# import torch
# from torch.utils import data
import glob
# import cv2
import SimpleITK as sitk



class DataLoader():
    def __init__(self, path_data):      
        path_data = os.path.normpath(path_data)
        self.pat_list = glob.glob(os.path.normpath( path_data + r"\rawdata\*"))
        self.n_pat = []
        self.data_list = []
        self.mask_list = []
        self.slice = []
        # print(self.pat_list[0])
        # self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
        # print(' '.join(self.name[0]))
        
        for i in range(2):
            path = [os.path.normpath(pat_list[i])]
            ind = np.random.permutation(np.arange(0,100))
            # file_reader = sitk.ImageFileReader()
            # file_reader.SetFileName(file_name)
            # file_reader.ReadImageInformation()
            # size=file_reader.GetSize()
            for k in range(20):
                self.data_list = self.data_list + path 
                self.mask_list = self.mask_list + [path[0].replace('rawdata','derivates',1)]
                self.slice = self.slice + [ind[k]]
                self.n_pat = self.n_pat + [i]
                # print(' '.join(self.data_list[i]))
                # print(self.path)
            
    # def __getitem__(self, index):
    #     img, H = medpy.io.load(self.data_list[index])
    #     # img = img[:,:,100]
    #     # img = cv2.resize(img, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
    #     img = torch.tensor(((img)+1024)/4096)

    #     mask, H = medpy.io.load(self.mask_list[index])
    #     # mask = mask[:,:,100]
    #     # mask = cv2.resize(mask, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
    #     mask = torch.tensor(mask.astype(np.bool_))
        
    #     return img, mask

loader = DataLoader(path_data = "C:\Data\Verse2019")


###### testovani funkcnosti
# img, mask = loader[0]

# plt.figure()
# plt.imshow(img[:,:,105],cmap='gray')
# plt.figure()
# plt.imshow(mask[:,:,105],cmap='gray')


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