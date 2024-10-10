## color
#%% librairies
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

#%% image
im =cv2.imread('skin_lesion_dataset/melanoma/ISIC_0000030.jpg',cv2.IMREAD_COLOR)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

red,green,blue = cv2.split(im)


subplot(231)
imshow(red,cmap=cm.gray)
subplot(232)
imshow(green,cmap=cm.gray)
subplot(233)
imshow(blue,cmap=cm.gray)

image_nb = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY )
subplot(234)
imshow(image_nb,cmap = cm.gray)
subplot(235)
imshow(im)


