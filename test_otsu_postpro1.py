#Affiche le masque binaire de segmentation pour otsu + post-processing 1

#Paramètre de l'élément structurant disque 
i_disk = 20

from otsu_seg import otsu
from post_processing import overall, opening
from display_image import mask_display
import skimage.morphology as morph
from find_central_component import find_largest_connected_component



def display_otsu_postpro1(img,i,j,k):
    tresh = otsu(img,None)
    struct_element1 = morph.disk(i)
    struct_element2 = morph.rectangle(j, k)
    new_image = find_largest_connected_component(mask_display(opening(img, morph.disk(i_disk)),None,tresh))
    return new_image

