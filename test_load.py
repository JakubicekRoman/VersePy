import load_data
# import matplotlib
import matplotlib.pyplot as plt

path_data = "C:/Data\Verse2019/rawdata/sub-verse004"

slice = 50;
# extract_size = [0, 0, 1] # for sagital view
# current_index = [-1, -1, slice] # for sagital view
extract_size = [0, 1, 0] # for axial view
current_index = [-1, slice, -1] # for axial view
# extract_size = [1, 0, 0] # for axial view
# current_index = [slice, -1, -1] # for axial view
img = load_data.read_nii_position(path_data + "/" + "sub-verse004_ct.nii.gz", extract_size,current_index )

plt.figure()
plt.imshow(img,cmap="gray")


