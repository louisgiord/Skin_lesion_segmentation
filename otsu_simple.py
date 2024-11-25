# Test file to display segmentation with Otsu's method without pre or post post-processing
from display_image import mask_display
from otsu_seg import otsu
from blk_removal import mask_remove

def display_otsu_simple(img):
    tresh = otsu(img,None)
    new_img = mask_display(img,None,tresh)
    return new_img
