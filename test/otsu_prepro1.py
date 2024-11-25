from otsu_seg import otsu
from post_processing import mask_remove
from display_image import mask_display

def display_otsu_prepro1(img, tau,l,x,y):
    mask = mask_remove(img,tau,l,x,y)
    tresh = otsu(img,mask)
    new_image = mask_display(img,mask,tresh)
    return new_image

