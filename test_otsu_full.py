from otsu_seg import otsu
from post_processing import overall, opening
from display_image import mask_display
from blk_removal import mask_remove
import skimage.morphology as morph
from find_central_component import find_largest_connected_component

def display_otsu_full(img,tau,l,x,y,i,j,k):
    mask = mask_remove(img,tau,l,x,y)
    tresh = otsu(img,mask)
    struct_element1 = morph.disk(i)
    struct_element2 = morph.rectangle(j, k)
    new_image = find_largest_connected_component(mask_display(opening(img,morph.disk(20)),mask,tresh))
    return new_image