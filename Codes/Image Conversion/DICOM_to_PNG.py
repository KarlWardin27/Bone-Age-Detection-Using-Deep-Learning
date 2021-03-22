import cv2
import os
import pydicom 

inputdir = 'C:/New folder/'
outdir = 'D:/Code_Me/Python/Bone_Age_Detection'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list: 
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.PixelData   # get image array
    cv2.imwrite(f.replace('.DCM','.png'),img)
    #ds.save_as("Try.png")