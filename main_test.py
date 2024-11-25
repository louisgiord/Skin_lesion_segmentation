import numpy as np 
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Markdown
import os
import sys 


from config import DICE as D
from test import display_otsu_simple 



# Parameters 

tau = 150
l = 5
x,y = 10,10
i,j,k= 40,10,3

# Load the images

img1 = cv2.imread("images_test/img1.jpg")
mask1 = cv2.imread("images_test/msk1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images_test/img2.jpg")
mask2 = cv2.imread("images_test/msk2.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("images_test/img3.jpg")
mask3 = cv2.imread("images_test/msk3.jpg", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("images_test/img4.jpg")
mask4 = cv2.imread("images_test/msk4.jpg", cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread("images_test/img5.jpg")
mask5 = cv2.imread("images_test/msk5.jpg", cv2.IMREAD_GRAYSCALE)
img6 = cv2.imread("images_test/img6.jpg")
mask6 = cv2.imread("images_test/msk6.jpg", cv2.IMREAD_GRAYSCALE)
img7 = cv2.imread("images_test/img7.jpg")
mask7 = cv2.imread("images_test/msk7.jpg", cv2.IMREAD_GRAYSCALE)
img8 = cv2.imread("images_test/img8.jpg")
mask8 = cv2.imread("images_test/msk8.jpg", cv2.IMREAD_GRAYSCALE)
img9 = cv2.imread("images_test/img9.jpg")
mask9 = cv2.imread("images_test/msk9.jpg", cv2.IMREAD_GRAYSCALE)
img10 = cv2.imread("images_test/img10.jpg")
mask10 = cv2.imread("images_test/msk10.jpg", cv2.IMREAD_GRAYSCALE)
img11 = cv2.imread("images_test/img11.jpg")
mask11 = cv2.imread("images_test/msk11.jpg", cv2.IMREAD_GRAYSCALE)
img12 = cv2.imread("images_test/img12.jpg")
mask12 = cv2.imread("images_test/msk12.jpg", cv2.IMREAD_GRAYSCALE)
img13 = cv2.imread("images_test/img13.jpg")
mask13 = cv2.imread("images_test/msk13.jpg", cv2.IMREAD_GRAYSCALE)
img14 = cv2.imread("images_test/img14.jpg")
mask14 = cv2.imread("images_test/msk14.jpg", cv2.IMREAD_GRAYSCALE)
img15 = cv2.imread("images_test/img15.jpg")
mask15 = cv2.imread("images_test/msk15.jpg", cv2.IMREAD_GRAYSCALE)
img16 = cv2.imread("images_test/img16.jpg")
mask16 = cv2.imread("images_test/msk16.jpg", cv2.IMREAD_GRAYSCALE)
img17 = cv2.imread("images_test/img17.jpg")
mask17 = cv2.imread("images_test/msk17.jpg", cv2.IMREAD_GRAYSCALE)
img18 = cv2.imread("images_test/img18.jpg")
mask18 = cv2.imread("images_test/msk18.jpg", cv2.IMREAD_GRAYSCALE)
img19 = cv2.imread("images_test/img19.jpg")
mask19 = cv2.imread("images_test/msk19.jpg", cv2.IMREAD_GRAYSCALE)
img20 = cv2.imread("images_test/img20.jpg")
mask20 = cv2.imread("images_test/msk20.jpg", cv2.IMREAD_GRAYSCALE)

# Image 1
img1_simple = os.display_otsu_simple(img1)
img1_pre1 = opre.display_otsu_prepro1(img1, tau, l, x, y)
img1_postpro1 = opost. display_otsu_postpro1(img1, i, j, k)
img1_full = ofull.display_otsu_full(img1, tau, l, x, y, i, j, k)

dice1 = D.dice(mask1, img1_simple)
dice1_pp1 = D.dice(mask1, img1_pre1)
dice1_pp = D.dice(mask1, img1_postpro1)
dice1_full = D.dice(mask1, img1_full)

# Image 2
img2_simple = os.display_otsu_simple(img2)
img2_pre1 = opre.display_otsu_prepro1(img2, tau, l, x, y)
img2_postpro1 = opost.display_otsu_postpro1(img2, i, j, k)
img2_full = ofull.display_otsu_full(img2, tau, l, x, y, i, j, k)

dice2 = D.dice(mask2, img2_simple)
dice2_pp1 = D.dice(mask2, img2_pre1)
dice2_pp = D.dice(mask2, img2_postpro1)
dice2_full = D.dice(mask2, img2_full)

# Image 3
img3_simple = os.display_otsu_simple(img3)
img3_pre1 = opre.display_otsu_prepro1(img3, tau, l, x, y)
img3_postpro1 = opost.display_otsu_postpro1(img3, i, j, k)
img3_full = ofull.display_otsu_full(img3, tau, l, x, y, i, j, k)

dice3 = D.dice(mask3, img3_simple)
dice3_pp1 = D.dice(mask3, img3_pre1)
dice3_pp = D.dice(mask3, img3_postpro1)
dice3_full = D.dice(mask3, img3_full)

# Image 4
img4_simple = os.display_otsu_simple(img4)
img4_pre1 = opre.display_otsu_prepro1(img4, tau, l, x, y)
img4_postpro1 = opost.display_otsu_postpro1(img4, i, j, k)
img4_full = ofull.display_otsu_full(img4, tau, l, x, y, i, j, k)

dice4 = D.dice(mask4, img4_simple)
dice4_pp1 = D.dice(mask4, img4_pre1)
dice4_pp = D.dice(mask4, img4_postpro1)
dice4_full = D.dice(mask4, img4_full)

# Image 5
img5_simple = os.display_otsu_simple(img5)
img5_pre1 = opre.display_otsu_prepro1(img5, tau, l, x, y)
img5_postpro1 = opost.display_otsu_postpro1(img5, i, j, k)
img5_full = ofull.display_otsu_full(img5, tau, l, x, y, i, j, k)

dice5 = D.dice(mask5, img5_simple)
dice5_pp1 = D.dice(mask5, img5_pre1)
dice5_pp = D.dice(mask5, img5_postpro1)
dice5_full = D.dice(mask5, img5_full)

# Image 6
img6_simple = os.display_otsu_simple(img6)
img6_pre1 = opre.display_otsu_prepro1(img6, tau, l, x, y)
img6_postpro1 = opost.display_otsu_postpro1(img6, i, j, k)
img6_full = ofull.display_otsu_full(img6, tau, l, x, y, i, j, k)

dice6 = D.dice(mask6, img6_simple)
dice6_pp1 = D.dice(mask6, img6_pre1)
dice6_pp = D.dice(mask6, img6_postpro1)
dice6_full = D.dice(mask6, img6_full)



D1 = [dice1, dice1_pp1, dice1_pp, dice1_full]
D2 = [dice2, dice2_pp1, dice2_pp, dice2_full]
D3 = [dice3, dice3_pp1, dice3_pp, dice3_full]
D4 = [dice4, dice4_pp1, dice4_pp, dice4_full]
D5 = [dice5, dice5_pp1, dice5_pp, dice5_full]
D6 = [dice6, dice6_pp1, dice6_pp, dice6_full]
"""
D7 = [dice7, dice7_pp1, dice7_pp, dice7_full]
D8 = [dice8, dice8_pp1, dice8_pp, dice8_full]
D9 = [dice9, dice9_pp1, dice9_pp, dice9_full]
D10 = [dice10, dice10_pp1, dice10_pp, dice10_full]
D11 = [dice11, dice11_pp1, dice11_pp, dice11_full]
D12 = [dice12, dice12_pp1, dice12_pp, dice12_full]
D13 = [dice13, dice13_pp1, dice13_pp, dice13_full]
D14 = [dice14, dice14_pp1, dice14_pp, dice14_full]
D15 = [dice15, dice15_pp1, dice15_pp, dice15_full]
D16 = [dice16, dice16_pp1, dice16_pp, dice16_full]
D17 = [dice17, dice17_pp1, dice17_pp, dice17_full]
D18 = [dice18, dice18_pp1, dice18_pp, dice18_full]
D19 = [dice19, dice19_pp1, dice19_pp, dice19_full]
D20 = [dice20, dice20_pp1, dice20_pp, dice20_full]"""

D = [D1, D2, D3, D4, D5, D6]#D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20]

def table_score (D):
    result1 = ["Image 1",D[0][0], D[0][1], D[0][2],D[0][3]]
    result2 = ["Image 2",D[1][0], D[1][1], D[1][2],D[1][3]]
    result3 = ["Image 3",D[2][0], D[2][1], D[2][2],D[2][3]]
    result4 = ["Image 4",D[3][0], D[3][1], D[3][2],D[3][3]]
    result5 = ["Image 5",D[4][0], D[4][1], D[4][2],D[4][3]]
    result6 = ["Image 6",D[5][0], D[5][1], D[5][2],D[5][3]]
    """
    result7 = ["Image 7",D[6][0], D[6][1], D[6][2],D[6][3]]
    result8 = ["Image 8",D[7][0], D[7][1], D[7][2],D[7][3]]
    result9 = ["Image 9",D[8][0], D[8][1], D[8][2],D[8][3]]
    result10 = ["Image 10",D[9][0], D[9][1], D[9][2],D[9][3]]
    result11 = ["Image 11",D[10][0], D[10][1], D[10][2],D[10][3]]
    result12 = ["Image 12",D[11][0], D[11][1], D[11][2],D[11][3]]
    result13 = ["Image 13",D[12][0], D[12][1], D[12][2],D[12][3]]
    result14 = ["Image 14",D[13][0], D[13][1], D[13][2],D[13][3]]
    result15 = ["Image 15",D[14][0], D[14][1], D[14][2],D[14][3]]
    result16 = ["Image 16",D[15][0], D[15][1], D[15][2],D[15][3]]
    result17 = ["Image 17",D[16][0], D[16][1], D[16][2],D[16][3]]
    result18 = ["Image 18",D[17][0], D[17][1], D[17][2],D[17][3]]
    result19 = ["Image 19",D[18][0], D[18][1], D[18][2],D[18][3]]
    result20 = ["Image 20",D[19][0], D[19][1], D[19][2],D[19][3]]"""
    results = [result1, result2, result3, result4, result5, result6]#, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20]



    table = "| Image | Otsu | Otsu + Pre-processing 1 | Otsu + Post-processing 1 | Otsu + Pre-processing 1 + Post-processing 1 \n"
    table += "| --- | --- | --- | --- | --- \n"
    for result in results:
        table += "| " + " | ".join([str(r) for r in result]) + " |\n"

    display(Markdown(table))