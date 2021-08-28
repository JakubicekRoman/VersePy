## imports

# libraries
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
from data_utilities import *
from glob import glob
import SimpleITK as sitk

## paths

# data directory
# directoryRaw = "C:\Data\Verse2019\rawdata"
# directionDer = "C:\Data\Verse2019\derivatives"
# pat = "sub-verse417"

data_list = glob("C:/Data/Verse2019/rawdata/*/*.nii.gz")
mask_list = glob("C:/Data/Verse2019/derivatives/*/*.nii.gz")
ctd_lists = glob("C:/Data/Verse2019/derivatives/*/*.json")

path_save = "D:/Python_Verse/data_reload"


# for i in [128]:
# for i in range(len(data_list)):
for i in range(129,160):

    # load files
    img_nib = nib.load(os.path.join(data_list[i]))
    msk_nib = nib.load(os.path.join(mask_list[i]))
    ctd_list = load_centroids(os.path.join(ctd_lists[i]))
    
    
    #check img zooms 
    zooms = img_nib.header.get_zooms()
    print('img zooms = {}'.format(zooms))
    
    #check img orientation
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
    print('img orientation code: {}'.format(axs_code))
    
    #check centroids
    print('Centroid List: {}'.format(ctd_list))
    
    
    # Resample and Reorient data
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    ctd_iso = rescale_centroids(ctd_list, img_nib, (1,1,1))
    
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))
    ctd_iso = reorient_centroids_to(ctd_iso, img_iso)
    
    #check img zooms 
    zooms = img_iso.header.get_zooms()
    print('img zooms = {}'.format(zooms))
    
    #check img orientation
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_iso.affine))
    print('img orientation code: {}'.format(axs_code))
    
    #check centroids
    print('new centroids: {}'.format(ctd_iso))
    
    # get vocel data
    im_np  = img_iso.get_fdata()
    msk_np = msk_iso.get_fdata()
    
    plt.figure()
    plt.imshow(im_np[:,:,20],cmap="gray")
    
    
    imNew = sitk.GetImageFromArray(im_np)
    maskNew = sitk.GetImageFromArray(msk_np)

    ### save nii new
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(os.path.join(data_list[i]))
    file_reader.ReadImageInformation()
    size = file_reader.GetSize()

    # for k in file_reader.GetMetaDataKeys(): 
    # #     imNew.SetMetaData(k,file_reader.GetMetaData(k))
    # #     maskNew.SetMetaData(k,file_reader.GetMetaData(k))
    #     v = file_reader.GetMetaData(k) 
    #     print("({0}) = = \"{1}\"".format(k,v))
          
    imNew.SetMetaData('dim[1]', str(im_np.shape[0]))
    imNew.SetMetaData('dim[2]', str(im_np.shape[1]))
    imNew.SetMetaData('dim[3]', str(im_np.shape[2]))
    
    maskNew.SetMetaData('dim[1]', str(im_np.shape[0]))
    maskNew.SetMetaData('dim[2]', str(im_np.shape[1]))
    maskNew.SetMetaData('dim[3]', str(im_np.shape[2]))

    
    num = "000" + str(i)
    num = num[-3:]
    name = "pat_" + num + "_raw.nii.gz"
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path_save + "/" + name)
    writer.Execute(imNew)
    
    name = "pat_" + num + "_mask" + ".nii.gz"
    writer.SetFileName(path_save + "/" + name)
    writer.Execute(maskNew)
    
    name = "pat_" + num + "_centroids" + ".json"
    save_centroids(ctd_iso, path_save + "/" + name)
    
    
    ##### get the mid-slice of the scan and mask in both sagittal and coronal planes
    
    # im_np_sag = im_np[:,:,int(im_np.shape[2]/2)]
    # im_np_cor = im_np[:,int(im_np.shape[1]/2),:]
    
    # msk_np_sag = msk_np[:,:,int(msk_np.shape[2]/2)]
    # msk_np_sag[msk_np_sag==0] = np.nan
    
    # msk_np_cor = msk_np[:,int(msk_np.shape[1]/2),:]
    # msk_np_cor[msk_np_cor==0] = np.nan
    
    # # plot 
    # fig, axs = create_figure(96,im_np_sag, im_np_cor)
    
    # axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
    # axs[0].imshow(msk_np_sag, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
    # plot_sag_centroids(axs[0], ctd_iso, zooms)
    
    # axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
    # axs[1].imshow(msk_np_cor, cmap=cm_itk, alpha=0.3, vmin=1, vmax=64)
    # plot_cor_centroids(axs[1], ctd_iso, zooms)
