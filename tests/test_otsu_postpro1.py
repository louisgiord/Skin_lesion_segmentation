# Display the binary segmentation mask for otsu + post-processing 1

# Parameter for the disk structuring element
i_disk = 20

from src.segmentation.otsu_seg import otsu
from src.postprocessing.post_processing import overall, opening
from src.utils.display_image import mask_display
import skimage.morphology as morph
from src.postprocessing.find_central_component import find_largest_connected_component



def display_otsu_postpro1(img,i,j,k):
    tresh = otsu(img,None)
    struct_element1 = morph.disk(i)
    struct_element2 = morph.rectangle(j, k)
    new_image = find_largest_connected_component(mask_display(opening(img, morph.disk(i_disk)),None,tresh))
    return new_image

