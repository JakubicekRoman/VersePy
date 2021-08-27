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
# import cv2
import SimpleITK as sitk


path_data = "C:\Data\Verse2019"

path_data = os.path.normpath(path_data)
pat_list = glob.glob(os.path.normpath( path_data + r"\rawdata\*"))
n_pat = []
data_list = []
mask_list = []
slice = []
# print(self.pat_list[0])
# self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
# print(' '.join(self.name[0]))

for i in range(1,2):
    path = os.path.normpath(pat_list[i])
    name = path.split(os.sep)
    name = name[-1]
    path_current = [path + os.sep + name + "_ct.nii.gz"] 
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(path_current[0])
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()    
    
    ind = np.random.permutation(np.arange(0,100)) 

    for k in range(20):
        data_list = data_list + path_current
        p = path_current[0].replace('rawdata','derivates',1)
        p = p.replace("_ct","_seg-vert_msk",1)
        mask_list = mask_list + [p]
        slice = slice + [ind[k]]
        n_pat = n_pat + [i]
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








