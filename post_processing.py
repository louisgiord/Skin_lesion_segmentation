#ensemble des fonctions du post-processing sans la fonction de recherche de la composante centrale

import skimage.morphology as morph
import cv2
from DICE import dice 

struct_element1 = morph.disk(3)

def dilate(img, struct_element):
    return morph.dilation(img, struct_element)

struct_element2 = morph.rectangle(2, 5)

def erode(img, struct_element):
    return morph.erosion(img, struct_element)

def overall(img, struct_element1, struct_element2):
    return erode(dilate(img, struct_element2), struct_element1)

def opening(img,struct_element):
    return morph.opening(img, struct_element)

def max_opening_dice(img,mask):
    max_dice = 0 
    i_max = 0 
    for i in range (30,40):
        new_mask = opening(img, morph.disk(i))
        dice_new = dice(new_mask, mask)
        if dice_new > max_dice:
            max_dice = dice_new
            i_max = i
    return i_max

