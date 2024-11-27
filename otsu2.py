import numpy as np 
import cv2
import matplotlib.pyplot as plt
from display_image import viewimgs, viewimage, mask_display
from blk_removal import mask_remove
from otsu_seg import otsu
from DICE import dice
from find_central_component import find_largest_connected_component
from post_processing import overall, opening
from skimage import morphology as morph

img = cv2.imread("images_test/img15.jpg")
mask = cv2.cvtColor(cv2.imread("images_test/msk15.png"), cv2.COLOR_BGR2GRAY)

tau = 150
x,y = 20,20
l = 5
i =20


# Otsu advanced thresholding

def display_otsu_level(img,tau,l,x,y,i):
    img1 = img[:,:,0]
    img2 = img[:,:,1]
    img3 = img[:,:,2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = mask_remove(img_gray,tau,l,x,y)
    mask1 = mask_remove(img1,tau,l,x,y)
    mask2 = mask_remove(img2,tau,l,x,y)
    mask3 = mask_remove(img3,tau,l,x,y)
    tresh = otsu(img_gray,mask)
    tresh1 = otsu(img1,mask1)
    tresh2 = otsu(img2,mask2)
    tresh3 = otsu(img3,mask3)
    res = mask_display(img_gray,mask,tresh)
    res1 = mask_display(img1,mask1,tresh1)
    res2 = mask_display(img2,mask2,tresh2)
    res3 = mask_display(img3,mask3,tresh3)
    final = find_largest_connected_component(opening(cv2.bitwise_or(res, cv2.bitwise_or(res1, cv2.bitwise_or(res2, res3))),morph.disk(i)))
    return final
