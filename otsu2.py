import numpy as np 
import cv2
import matplotlib.pyplot as plt
from display_image import viewimgs, viewimage, mask_display
from blk_removal import mask_remove
from otsu_seg import otsu

img = cv2.imread("images_test/img1.jpg")

tau = 60
x,y = 20,20
l = 5

img1 = img[:,:,2] #red channel
img2 = img[:,:,1] #green channel
img3 = img[:,:,0] #blue channel

# Otsu advanced thresholding

            

def double_mask(img,tau,l,x,y):
    Lx = img.shape[1]
    Ly = img.shape[0]
    mask = mask_remove(img,tau,l,x,y)
    tresh = otsu(img,mask)
    new_mask = cv2.threshold(img, tresh, 1, cv2.THRESH_BINARY)[1]
    return new_mask

def apply_threshold_with_mask(img, mask, thresh, maxval, type):
    # Appliquer le masque pour restreindre la région de l'image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Appliquer le seuillage sur la région masquée de l'image
    retval, thresholded_img = cv2.threshold(masked_img, thresh, maxval, type)
    
    # Combiner l'image seuillée avec les parties non masquées de l'image originale
    result = cv2.bitwise_or(thresholded_img, cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)))
    
    return result
            

def otsu_level (img,tau,l,x,y):
    mask = double_mask(img,tau,l,x,y)
    viewimage(mask*150)
    tresh = otsu(img,mask)
    print(tresh)
    new_img = mask_display(img,mask,tresh)
    return new_img

#viewimage(double_mask(img2,tau,l,x,y)*255)
new_img = otsu_level(img2,tau,l,x,y)
viewimage(new_img)
mask = mask_remove(img2,tau,l,x,y)
tresh = otsu(img1,mask)
im_= mask_display(img1,mask,tresh)
viewimage(cv2.bitwise_or(new_img,im_)) 