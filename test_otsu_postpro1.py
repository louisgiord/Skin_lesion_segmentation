from otsu_seg import otsu
from post_processing import overall, closing, max_closing_dice
from display_image import mask_display
import skimage.morphology as morph



def display_otsu_postpro1(img,i,j,k):
    tresh = otsu(img,None)
    struct_element1 = morph.disk(i)
    struct_element2 = morph.rectangle(j, k)
    new_image = mask_display(closing(img,morph.disk(5)),None,tresh)
    return new_image

