#%%
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import blk_removal as blk
import otsu_seg as os 
import post_processing as pp
from display_image import viewimgs, viewimage
#%%
#Download of the test images 
img1 = cv2.imread('images_test/img1.jpg')
img2 = cv2.imread('images_test/img8.jpg')
#%%
#Test images in grayscale

img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#%%
mask = blk.Remove(img_gray1, 60, 5, 5, 5)
viewimage(mask)

 #%%
#Test of the region growing fonction



# %%
tresh = os.otsu(img_gray1)
mask = cv2.threshold(img_gray1, tresh, 255, cv2.THRESH_BINARY_INV)[1]
viewimage(mask)

# %%
