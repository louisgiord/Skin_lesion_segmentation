import skimage.morphology as morph
from DICE import dice
import cv2
from otsu_seg import otsu
from display_image import mask_display
from blk_removal import mask_remove

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


def max_struct_element(img,mask):
    dice_max = 0 
    i_max = 0
    j_max = 0
    k_max = 0
    for i in range (1,30):
        for j in range (1,20):
            for k in range (1,20):
                struct_element1 = morph.disk(i)
                struct_element2 = morph.rectangle(j, k)
                img_pp = overall(img, struct_element1, struct_element2)
                dice_score = dice(img_pp, mask)
                if dice_score > dice_max:
                    dice_max = dice_score
                    i_max = i
                    j_max = j
                    k_max = k
    return i_max,j_max,k_max

