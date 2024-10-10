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

