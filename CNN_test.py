import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt
# import matplotlib.image as im
# import nibabel as nib

import SimpleITK as sitk
# import medpy as medpy
# import pandas as pd
import torch
from torch.utils import data
import glob
# import cv2
import SimpleITK as sitk
import load_data


class DataLoader():
    def __init__(self, path_data, pat):      
        path_data = os.path.normpath(path_data)
        self.pat_list = glob.glob(os.path.normpath( path_data + "\*raw.nii.gz"))
        n_pat = []
        self.data_list = []
        self.mask_list = []
        self.slice = []
        self.n_pat = []
        self.size_data=[]
        # print(self.pat_list[0])
        # self.name = glob.glob(self.pat_list[0] + "/*nii.gz")
        # print(' '.join(self.name[0]))
        
        for i in pat:
            path = os.path.normpath(self.pat_list[i])
            
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(path)
            file_reader.ReadImageInformation()
            size=file_reader.GetSize()
    
            ind = np.random.permutation(np.arange(20,size[2])) 
            
            for k in range(10):
                # for k in [0]:
                self.data_list = self.data_list + [os.path.normpath(path)]
                p =  os.path.normpath(path.replace("raw","mask",1))
                self.mask_list = self.mask_list + [p]
                self.slice = self.slice + [ind[k]]
                self.n_pat = self.n_pat + [i]
                self.size_data = self.size_data + [size]
                # print(' '.join(self.data_list[i]))
                # print(self.path)
                
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        size = self.size_data[index]
        size_cut = [224,224,1];
        vel_maxTr = [np.maximum(size[0]-size_cut[0]-1,1), np.maximum(size[1]-size_cut[1]-1,1)]
        transl = [np.random.randint(0,vel_maxTr[0]), np.random.randint(0,vel_maxTr[1]), int(self.slice[index])]

        img = load_data.read_nii_position(self.data_list[index], size_cut, transl)
        img = torch.tensor(((img)+1024)/4096)

        size_cut = [224,224,1];
        mask = load_data.read_nii_position(self.mask_list[index], size_cut, transl)
        mask[mask<0]=0
        mask[mask>0]=255
        mask = torch.tensor(mask.astype(np.bool_))
        
        return img, mask


# training loader
loader = DataLoader(path_data = "C:\Data\Verse2019\data_reload", pat=range(0,5))
trainloader= data.DataLoader(loader,batch_size=2, num_workers=0, shuffle=True, drop_last=True)

# testing loader
loader = DataLoader(path_data = "C:\Data\Verse2019\data_reload", pat=range(5,7))
testloader= data.DataLoader(loader,batch_size=2, num_workers=0, shuffle=False, drop_last=True)



###### testovani funkcnosti
# img, mask = loader[3]

# plt.figure()
# plt.imshow(img,cmap='gray')
# plt.figure()
# plt.imshow(mask,cmap='gray')


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