import skimage.morphology as morph
import cv2
from display_image import mask_display
from DICE import dice 
from otsu_seg import otsu




#load images 

img1 = cv2.imread("images_test/img2.jpg")
mask1 = cv2.imread("images_test/mask2.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images_test/img3.jpg")
mask2 = cv2.imread("images_test/mask3.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("images_test/img4.jpg")
mask3 = cv2.imread("images_test/mask4.jpg", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("images_test/img5.jpg")
mask4 = cv2.imread("images_test/mask5.jpg", cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread("images_test/img6.jpg")
mask5 = cv2.imread("images_test/mask6.jpg", cv2.IMREAD_GRAYSCALE)
img6 = cv2.imread("images_test/img7.jpg")
mask6 = cv2.imread("images_test/mask7.jpg", cv2.IMREAD_GRAYSCALE)
img7 = cv2.imread("images_test/img8.jpg")
mask7 = cv2.imread("images_test/mask8.jpg", cv2.IMREAD_GRAYSCALE)
img8 = cv2.imread("images_test/img9.jpg")
mask8 = cv2.imread("images_test/mask9.jpg", cv2.IMREAD_GRAYSCALE)
img9 = cv2.imread("images_test/img10.jpg")
mask9 = cv2.imread("images_test/mask10.jpg", cv2.IMREAD_GRAYSCALE)
img10 = cv2.imread("images_test/img11.jpg")
mask10 = cv2.imread("images_test/mask11.jpg", cv2.IMREAD_GRAYSCALE)

struct_element1 = morph.disk(3)

def dilate(img, struct_element):
    return morph.dilation(img, struct_element)

struct_element2 = morph.rectangle(2, 5)

def erode(img, struct_element):
    return morph.erosion(img, struct_element)

def overall(img, struct_element1, struct_element2):
    return dilate(erode(img, struct_element2), struct_element1)


def max_struct_element1(img,mask):
    dice_max = 0 
    i_max = 0
    struct_element2 = morph.rectangle(2, 5)
    for i in range (1,30):
        struct_element1 = morph.disk(i)
        tresh = otsu(img)
        mask1 = overall(mask_display(img, None,tresh), struct_element1, struct_element2)
        dice_val = dice(mask, mask1)
        if dice_val > dice_max:
            dice_max = dice_val
            i_max = i
    return i_max

