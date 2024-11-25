import numpy as np 
import cv2
import matplotlib.pyplot as plt
from display_image import viewimgs, viewimage, mask_display
from blk_removal import mask_remove
from otsu_seg import otsu

img = cv2.imread("images_test/img10.jpg")

tau = 150
x,y = 20,20
l = 5

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    tresh = otsu(img,mask)
    print(tresh)
    new_img = mask_display(img,mask,tresh)
    return new_img
