import numpy as np
import SimpleITK as sitk


def read_raw_position(file_name,extract_size,current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()
    
    
    for k in range(3):
        if current_index[k]==-1:
            current_index[k]=0
            extract_size[k]=size[k]
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    img=np.zeros(extract_size,dtype=np.float32)
    
    tmp=sitk.GetArrayFromImage(file_reader.Execute())
    
    img[:tmp.shape[0],:tmp.shape[1],:tmp.shape[2]]=tmp
    
    return img



def read_nii_position(file_name,extract_size,current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()
        
    view = [1,2,0] # for sagital view
    # current_index = current_index[view[0], view[1], view[2]]
    # extract_size = extract_size[view[0], view[1], view[2]]
    
    for k in range(3):
        if current_index[k]==-1:
            current_index[k]=0
            extract_size[k]=size[k]
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
   
    img = sitk.GetArrayFromImage(file_reader.Execute())
    # img = np.transpose(img, (view[0], view[1], view[2])) # for sagital view
    img = np.squeeze(img)


    
    return img