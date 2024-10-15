### DULL RAZOR
#%% librairies

import numpy as np
"""
import platform
import tempfile
import os
from scipy import ndimage as ndi"""

import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.color import rgb2gray

import cv2

#%%passage en NB
im_color = skio.imread('skin_lesion_dataset/melanoma/ISIC_0000030.jpg')
im_grey = rgb2gray(im_color) #passer l'image en RGB > NB,
# choix d'une conversion plus contrastée et tenant compte de l'oeil humain ?

#%% opencv
image_col=cv2.imread('skin_lesion_dataset/melanoma/ISIC_0000030.jpg',cv2.IMREAD_COLOR)
image_nb = cv2.cvtColor(image_col, cv2.COLOR_RGB2GRAY )

red, green, blue = cv2.split(image_col)

#Black hat filter
kernel = cv2.getStructuringElement(1,(9,9)) 
blackhat_red = cv2.morphologyEx(red, cv2.MORPH_BLACKHAT, kernel)
blackhat_green = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
blackhat_blue = cv2.morphologyEx(blue, cv2.MORPH_BLACKHAT, kernel)

#Gaussian filter
bhg_red= cv2.GaussianBlur(blackhat_red,(3,3),cv2.BORDER_DEFAULT)
bhg_green= cv2.GaussianBlur(blackhat_green,(3,3),cv2.BORDER_DEFAULT)
bhg_blue= cv2.GaussianBlur(blackhat_blue,(3,3),cv2.BORDER_DEFAULT)

#Binary thresholding (MASK)
ret,mask = cv2.threshold(bhg_red,10,255,cv2.THRESH_BINARY)

#Replace pixels of the mask
dst = cv2.inpaint(image_col,mask,6,cv2.INPAINT_TELEA) 
