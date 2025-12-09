# Display the binary segmentation mask for otsu + pre-processing 1

from src.segmentation.otsu_seg import otsu
from src.preprocessing.blk_removal import mask_remove
from src.utils.display_image import mask_display

def display_otsu_prepro1(img, tau,l,x,y):
    mask = mask_remove(img,tau,l,x,y)
    tresh = otsu(img,mask)
    new_image = mask_display(img,mask,tresh)
    return new_image

