import numpy as np 
import cv2
import matplotlib.pyplot as plt
from display_image import viewimgs, viewimage, mask_display
from blk_removal import mask_remove
from otsu_seg import otsu
from DICE import dice

img = cv2.imread("images_test/img1.jpg")
mask = cv2.cvtColor(cv2.imread("images_test/msk1.png"), cv2.COLOR_BGR2GRAY)

tau = 150
x,y = 20,20
l = 5


# Otsu advanced thresholding

def display_otsu_level(img,tau,l,x,y):
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
    final = cv2.bitwise_or(res, cv2.bitwise_or(res1, cv2.bitwise_or(res2, res3)))
    return final

result = display_otsu_level(img,tau,l,x,y)
val_dice = dice(result,mask)
print("valeur dice :", val_dice)
viewimage(result)