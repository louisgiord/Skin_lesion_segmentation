from otsu_seg import otsu
from post_processing import overall
from display_image import mask_display
from blk_removal import mask_remove
import skimage.morphology as morph

def display_otsu_full(img,tau,l,x,y,i,j,k):
    mask = mask_remove(img,tau,l,x,y)
    tresh = otsu(img,mask)
    struct_element1 = morph.disk(i)
    struct_element2 = morph.rectangle(j, k)
    new_image = mask_display(overall(img, struct_element1, struct_element2),mask,tresh)
    return new_image