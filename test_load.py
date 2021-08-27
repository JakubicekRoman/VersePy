# import load_data
# import os
# # import matplotlib
# import matplotlib.pyplot as plt
# import SimpleITK as sitk

# path_data = "C:/Data/Verse2019/rawdata"
# path_masks = "C:/Data/Verse2019/derivatives"

# pat = "sub-verse05"

# file_reader = sitk.ImageFileReader()
# file_reader.SetFileName(path_data + "/" + pat + "/" + pat + "_ct.nii.gz")    
# file_reader.ReadImageInformation()
# size=file_reader.GetSize()

# slice = int(size[2]/2)

# extract_size = [0, 0, 1] # for sagital view
# current_index = [-1, -1, slice] # for sagital view
# # # current_index = [-1, -1, -1] # for sagital view
# # extract_size = [0, 0, 0] # for axial view
# # current_index = [-1, slice, -1] # for axial view
# # extract_size = [1, 0, 0] # for coronal view
# # current_index = [slice, -1, -1] # for coronal view

# img = load_data.read_nii_position(os.path.normpath(path_data + "/" + pat + "/" + pat + "_ct.nii.gz"), extract_size,current_index )

# # mask = load_data.read_nii_position(os.path.normpath(path_masks + "/" + pat + "/" + pat + "_seg-vert_msk.nii.gz"), extract_size,current_index )
# mask = load_data.read_nii_position(path_masks + "/" + pat + "/" + pat + "_seg-vert_msk.nii.gz", extract_size,current_index)

# plt.figure()
# plt.imshow(img,cmap="gray")

# plt.figure()
# plt.imshow(mask,cmap="gray")



#####


import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
# import matplotlib.image as im
# import nibabel as nib

# import medpy as medpy
# import pandas as pd
# import torch
# from torch.utils import data
import glob
import cv2
import SimpleITK as sitk
import load_data


# path_data = "C:\Data\Verse2019"
path_data = "D:\Python_Verse\data_reload"

path_data = os.path.normpath(path_data)
pat_list = glob.glob(os.path.normpath( path_data + "\*raw.nii.gz"))
n_pat = []
data_list = []
mask_list = []
slice = []
# print(self.pat_list[0])
# self.name = glob.glob(self.pat_list[0] + "/*nii.gz")cccc
# print(' '.join(self.name[0]))

# for i in range(1,2):
for i in [0]:
# for i in [1,2,3,4,5,6,7,8,10,11]:
    path = os.path.normpath(pat_list[i])
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(path)
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()
    
    ind = np.random.permutation(np.arange(0,size[0])) 

    # for k in range(20):
    for k in [0]:
        data_list = data_list + [os.path.normpath(path)]
        p =  os.path.normpath(path.replace("raw","mask",1))
        mask_list = mask_list + [p]
        slice = slice + [ind[k]]
        n_pat = n_pat + [i]
        # print(' '.join(self.data_list[i]))
        # print(self.path)
            
    # def __getitem__(self, index):

        img = load_data.read_nii_position(data_list[k],[0,0,1], [-1, -1, int(slice[k]) ])
        img = img[0:np.min(img.shape),0:np.min(img.shape)]
        img = cv2.resize(img, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
        
        mask = load_data.read_nii_position(mask_list[k],[0,0,1], [-1, -1, int(slice[k]) ])
        mask = mask[0:np.min(mask.shape),0:np.min(mask.shape)]
        mask = cv2.resize(mask, dsize=(124, 124), interpolation=cv2.INTER_NEAREST)    
        
        plt.figure()
        plt.imshow(img,cmap="gray")

        plt.figure()
        plt.imshow(mask,cmap="gray")
        
        # img = torch.tensor(((img)+1024)/4096)

        # mask, H = medpy.io.load(self.mask_list[index])
        # # mask = mask[:,:,100]
        # # mask = cv2.resize(mask, dsize=(124, 124), interpolation=cv2.INTER_CUBIC)
        # mask = torch.tensor(mask.astype(np.bool_))
        
        # return img, mask








