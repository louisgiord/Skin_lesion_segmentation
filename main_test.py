import numpy as np 
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Markdown
from test_otsu_full import *
from test_otsu_postpro1 import *
from test_otsu_prepro1 import *
from test_otsu_simple import *
from DICE import *
from display_image import *


# Parameters 

tau = 150
l = 5
x,y = 10,10
i,j,k= 30,3,10

# Load the images

img1 = cv2.cvtColor(cv2.imread("images_test/img1.jpg"), cv2.COLOR_BGR2GRAY)
mask1 = cv2.cvtColor(cv2.imread("images_test/msk1.png"),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("images_test/img2.jpg"), cv2.COLOR_BGR2GRAY)
mask2 = cv2.cvtColor(cv2.imread("images_test/msk2.png"),cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(cv2.imread("images_test/img3.jpg"), cv2.COLOR_BGR2GRAY)
mask3 = cv2.cvtColor(cv2.imread("images_test/msk3.png"),cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(cv2.imread("images_test/img4.jpg"), cv2.COLOR_BGR2GRAY)
mask4 = cv2.cvtColor(cv2.imread("images_test/msk4.png"),cv2.COLOR_BGR2GRAY)
img5 = cv2.cvtColor(cv2.imread("images_test/img5.jpg"), cv2.COLOR_BGR2GRAY)
mask5 = cv2.cvtColor(cv2.imread("images_test/msk5.png"),cv2.COLOR_BGR2GRAY)
img6 = cv2.cvtColor(cv2.imread("images_test/img6.jpg"), cv2.COLOR_BGR2GRAY)
mask6 = cv2.cvtColor(cv2.imread("images_test/msk6.png"),cv2.COLOR_BGR2GRAY)
img7 = cv2.cvtColor(cv2.imread("images_test/img7.jpg"), cv2.COLOR_BGR2GRAY)
mask7 = cv2.cvtColor(cv2.imread("images_test/msk7.png"),cv2.COLOR_BGR2GRAY)
img8 = cv2.cvtColor(cv2.imread("images_test/img8.jpg"), cv2.COLOR_BGR2GRAY)
mask8 = cv2.cvtColor(cv2.imread("images_test/msk8.png"),cv2.COLOR_BGR2GRAY)
img9 = cv2.cvtColor(cv2.imread("images_test/img9.jpg"), cv2.COLOR_BGR2GRAY)
mask9 = cv2.cvtColor(cv2.imread("images_test/msk9.png"),cv2.COLOR_BGR2GRAY)
img10 = cv2.cvtColor(cv2.imread("images_test/img10.jpg"), cv2.COLOR_BGR2GRAY)
mask10 = cv2.cvtColor(cv2.imread("images_test/msk10.png"),cv2.COLOR_BGR2GRAY)
img11 = cv2.cvtColor(cv2.imread("images_test/img11.jpg"), cv2.COLOR_BGR2GRAY)
mask11 = cv2.cvtColor(cv2.imread("images_test/msk11.png"),cv2.COLOR_BGR2GRAY)
img12 = cv2.cvtColor(cv2.imread("images_test/img12.jpg"), cv2.COLOR_BGR2GRAY)
mask12 = cv2.cvtColor(cv2.imread("images_test/msk12.png"),cv2.COLOR_BGR2GRAY)
img13 = cv2.cvtColor(cv2.imread("images_test/img13.jpg"), cv2.COLOR_BGR2GRAY)
mask13 = cv2.cvtColor(cv2.imread("images_test/msk13.png"),cv2.COLOR_BGR2GRAY)
img14 = cv2.cvtColor(cv2.imread("images_test/img14.jpg"), cv2.COLOR_BGR2GRAY)
mask14 = cv2.cvtColor(cv2.imread("images_test/msk14.png"),cv2.COLOR_BGR2GRAY)
img15 = cv2.cvtColor(cv2.imread("images_test/img15.jpg"), cv2.COLOR_BGR2GRAY)
mask15 = cv2.cvtColor(cv2.imread("images_test/msk15.png"),cv2.COLOR_BGR2GRAY)
img16 = cv2.cvtColor(cv2.imread("images_test/img16.jpg"), cv2.COLOR_BGR2GRAY)
mask16 = cv2.cvtColor(cv2.imread("images_test/msk16.png"),cv2.COLOR_BGR2GRAY)
img17 = cv2.cvtColor(cv2.imread("images_test/img17.jpg"), cv2.COLOR_BGR2GRAY)
mask17 = cv2.cvtColor(cv2.imread("images_test/msk17.png"),cv2.COLOR_BGR2GRAY)
img18 = cv2.cvtColor(cv2.imread("images_test/img18.jpg"), cv2.COLOR_BGR2GRAY)
mask18 = cv2.cvtColor(cv2.imread("images_test/msk18.png"),cv2.COLOR_BGR2GRAY)
img19 = cv2.cvtColor(cv2.imread("images_test/img19.jpg"), cv2.COLOR_BGR2GRAY)
mask19 = cv2.cvtColor(cv2.imread("images_test/msk19.png"),cv2.COLOR_BGR2GRAY)
img20 = cv2.cvtColor(cv2.imread("images_test/img20.jpg"), cv2.COLOR_BGR2GRAY)
mask20 = cv2.cvtColor(cv2.imread("images_test/msk20.png"),cv2.COLOR_BGR2GRAY)


# Image 1
img1_simple = display_otsu_simple(img1)
img1_pre1 = display_otsu_prepro1(img1, tau, l, x, y)
img1_postpro1 = display_otsu_postpro1(img1, i, j, k)
img1_full = display_otsu_full(img1, tau, l, x, y, i, j, k)

dice1 = dice(mask1, img1_simple)
dice1_pp1 = dice(mask1, img1_pre1)
dice1_pp = dice(mask1, img1_postpro1)
dice1_full = dice(mask1, img1_full)

# Image 2
img2_simple = display_otsu_simple(img2)
img2_pre1 = display_otsu_prepro1(img2, tau, l, x, y)
img2_postpro1 = display_otsu_postpro1(img2, i, j, k)
img2_full = display_otsu_full(img2, tau, l, x, y, i, j, k)

dice2 = dice(mask2, img2_simple)
dice2_pp1 = dice(mask2, img2_pre1)
dice2_pp = dice(mask2, img2_postpro1)
dice2_full = dice(mask2, img2_full)

# Image 3
img3_simple = display_otsu_simple(img3)
img3_pre1 = display_otsu_prepro1(img3, tau, l, x, y)
img3_postpro1 = display_otsu_postpro1(img3, i, j, k)
img3_full = display_otsu_full(img3, tau, l, x, y, i, j, k)

dice3 = dice(mask3, img3_simple)
dice3_pp1 = dice(mask3, img3_pre1)
dice3_pp = dice(mask3, img3_postpro1)
dice3_full = dice(mask3, img3_full)

# Image 4
img4_simple = display_otsu_simple(img4)
img4_pre1 = display_otsu_prepro1(img4, tau, l, x, y)
img4_postpro1 = display_otsu_postpro1(img4, i, j, k)
img4_full = display_otsu_full(img4, tau, l, x, y, i, j, k)

dice4 = dice(mask4, img4_simple)
dice4_pp1 = dice(mask4, img4_pre1)
dice4_pp = dice(mask4, img4_postpro1)
dice4_full = dice(mask4, img4_full)

# Image 5
img5_simple = display_otsu_simple(img5)
img5_pre1 = display_otsu_prepro1(img5, tau, l, x, y)
img5_postpro1 = display_otsu_postpro1(img5, i, j, k)
img5_full = display_otsu_full(img5, tau, l, x, y, i, j, k)

dice5 = dice(mask5, img5_simple)
dice5_pp1 = dice(mask5, img5_pre1)
dice5_pp = dice(mask5, img5_postpro1)
dice5_full = dice(mask5, img5_full)

# Image 6
img6_simple = display_otsu_simple(img6)
img6_pre1 = display_otsu_prepro1(img6, tau, l, x, y)
img6_postpro1 = display_otsu_postpro1(img6, i, j, k)
img6_full = display_otsu_full(img6, tau, l, x, y, i, j, k)

dice6 = dice(mask6, img6_simple)
dice6_pp1 = dice(mask6, img6_pre1)
dice6_pp = dice(mask6, img6_postpro1)
dice6_full = dice(mask6, img6_full)

# Image 7
img7_simple = display_otsu_simple(img7)
img7_pre1 = display_otsu_prepro1(img7, tau, l, x, y)
img7_postpro1 = display_otsu_postpro1(img7, i, j, k)
img7_full = display_otsu_full(img7, tau, l, x, y, i, j, k)

dice7 = dice(mask7, img7_simple)
dice7_pp1 = dice(mask7, img7_pre1)
dice7_pp = dice(mask7, img7_postpro1)
dice7_full = dice(mask7, img7_full)

# Image 8
img8_simple = display_otsu_simple(img8)
img8_pre1 = display_otsu_prepro1(img8, tau, l, x, y)
img8_postpro1 = display_otsu_postpro1(img8, i, j, k)
img8_full = display_otsu_full(img8, tau, l, x, y, i, j, k)

dice8 = dice(mask8, img8_simple)
dice8_pp1 = dice(mask8, img8_pre1)
dice8_pp = dice(mask8, img8_postpro1)
dice8_full = dice(mask8, img8_full)

# Image 9
img9_simple = display_otsu_simple(img9)
img9_pre1 = display_otsu_prepro1(img9, tau, l, x, y)
img9_postpro1 = display_otsu_postpro1(img9, i, j, k)
img9_full = display_otsu_full(img9, tau, l, x, y, i, j, k)

dice9 = dice(mask9, img9_simple)
dice9_pp1 = dice(mask9, img9_pre1)
dice9_pp = dice(mask9, img9_postpro1)
dice9_full = dice(mask9, img9_full)

# Image 10
img10_simple = display_otsu_simple(img10)
img10_pre1 = display_otsu_prepro1(img10, tau, l, x, y)
img10_postpro1 = display_otsu_postpro1(img10, i, j, k)
img10_full = display_otsu_full(img10, tau, l, x, y, i, j, k)

dice10 = dice(mask10, img10_simple)
dice10_pp1 = dice(mask10, img10_pre1)
dice10_pp = dice(mask10, img10_postpro1)
dice10_full = dice(mask10, img10_full)

# Image 11
img11_simple = display_otsu_simple(img11)
img11_pre1 = display_otsu_prepro1(img11, tau, l, x, y)
img11_postpro1 = display_otsu_postpro1(img11, i, j, k)
img11_full = display_otsu_full(img11, tau, l, x, y, i, j, k)

dice11 = dice(mask11, img11_simple)
dice11_pp1 = dice(mask11, img11_pre1)
dice11_pp = dice(mask11, img11_postpro1)
dice11_full = dice(mask11, img11_full)

# Image 12
img12_simple = display_otsu_simple(img12)
img12_pre1 = display_otsu_prepro1(img12, tau, l, x, y)
img12_postpro1 = display_otsu_postpro1(img12, i, j, k)
img12_full = display_otsu_full(img12, tau, l, x, y, i, j, k)

dice12 = dice(mask12, img12_simple)
dice12_pp1 = dice(mask12, img12_pre1)
dice12_pp = dice(mask12, img12_postpro1)
dice12_full = dice(mask12, img12_full)

# Image 13
img13_simple = display_otsu_simple(img13)
img13_pre1 = display_otsu_prepro1(img13, tau, l, x, y)
img13_postpro1 = display_otsu_postpro1(img13,i, j, k)
img13_full = display_otsu_full(img13, tau, l, x, y, i, j, k)

dice13 = dice(mask13, img13_simple)
dice13_pp1 = dice(mask13, img13_pre1)
dice13_pp = dice(mask13, img13_postpro1)
dice13_full = dice(mask13, img13_full)

# Image 14
img14_simple = display_otsu_simple(img14)
img14_pre1 = display_otsu_prepro1(img14, tau, l, x, y)
img14_postpro1 = display_otsu_postpro1(img14, i, j, k)
img14_full = display_otsu_full(img14, tau, l, x, y, i, j, k)

dice14 = dice(mask14, img14_simple)
dice14_pp1 = dice(mask14, img14_pre1)
dice14_pp = dice(mask14, img14_postpro1)
dice14_full = dice(mask14, img14_full)

# Image 15
img15_simple = display_otsu_simple(img15)
img15_pre1 = display_otsu_prepro1(img15, tau, l, x, y)
img15_postpro1 = display_otsu_postpro1(img15, i, j, k)
img15_full = display_otsu_full(img15, tau, l, x, y, i, j, k)

dice15 = dice(mask15, img15_simple)
dice15_pp1 = dice(mask15, img15_pre1)
dice15_pp = dice(mask15, img15_postpro1)
dice15_full = dice(mask15, img15_full)

# Image 16
img16_simple = display_otsu_simple(img16)
img16_pre1 = display_otsu_prepro1(img16, tau, l, x, y)
img16_postpro1 = display_otsu_postpro1(img16, i, j, k)
img16_full = display_otsu_full(img16, tau, l, x, y, i, j, k)

dice16 = dice(mask16, img16_simple)
dice16_pp1 = dice(mask16, img16_pre1)
dice16_pp = dice(mask16, img16_postpro1)
dice16_full = dice(mask16, img16_full)

# Image 17
img17_simple = display_otsu_simple(img17)
img17_pre1 = display_otsu_prepro1(img17, tau, l, x, y)
img17_postpro1 = display_otsu_postpro1(img17, i, j, k)
img17_full = display_otsu_full(img17, tau, l, x, y, i, j, k)

dice17 = dice(mask17, img17_simple)
dice17_pp1 = dice(mask17, img17_pre1)
dice17_pp = dice(mask17, img17_postpro1)
dice17_full = dice(mask17, img17_full)

# Image 18
img18_simple = display_otsu_simple(img18)
img18_pre1 = display_otsu_prepro1(img18, tau, l, x, y)
img18_postpro1 = display_otsu_postpro1(img18, i, j, k)
img18_full = display_otsu_full(img18, tau, l, x, y, i, j, k)

dice18 = dice(mask18, img18_simple)
dice18_pp1 = dice(mask18, img18_pre1)
dice18_pp = dice(mask18, img18_postpro1)
dice18_full = dice(mask18, img18_full)

# Image 19
img19_simple = display_otsu_simple(img19)
img19_pre1 = display_otsu_prepro1(img19, tau, l, x, y)
img19_postpro1 = display_otsu_postpro1(img19, i, j, k)
img19_full = display_otsu_full(img19, tau, l, x, y, i, j, k)

dice19 = dice(mask19, img19_simple)
dice19_pp1 = dice(mask19, img19_pre1)
dice19_pp = dice(mask19, img19_postpro1)
dice19_full = dice(mask19, img19_full)

# Image 20
img20_simple = display_otsu_simple(img20)
img20_pre1 = display_otsu_prepro1(img20, tau, l, x, y)
img20_postpro1 = display_otsu_postpro1(img20, i, j, k)
img20_full = display_otsu_full(img20, tau, l, x, y, i, j, k)

dice20 = dice(mask20, img20_simple)
dice20_pp1 = dice(mask20, img20_pre1)
dice20_pp = dice(mask20, img20_postpro1)
dice20_full = dice(mask20, img20_full)





D1 = [dice1, dice1_pp1, dice1_pp, dice1_full]
D2 = [dice2, dice2_pp1, dice2_pp, dice2_full]
D3 = [dice3, dice3_pp1, dice3_pp, dice3_full]
D4 = [dice4, dice4_pp1, dice4_pp, dice4_full]
D5 = [dice5, dice5_pp1, dice5_pp, dice5_full]
D6 = [dice6, dice6_pp1, dice6_pp, dice6_full]
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
D20 = [dice20, dice20_pp1, dice20_pp, dice20_full]

D = [D1, D2, D3, D4, D5, D6,D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20]


#Otsu_level 

i = 20

# Image 1
img1_color = cv2.imread("images_test/img1.jpg")
img1_lvl = display_otsu_level(img1_color,tau,l,x,y,i)
dice1_lvl = dice(mask1, img1_lvl)

# Image 2
img2_color = cv2.imread("images_test/img2.jpg")
img2_lvl = display_otsu_level(img2_color,tau,l,x,y,i)
dice2_lvl = dice(mask2, img2_lvl)

# Image 3
img3_color = cv2.imread("images_test/img3.jpg")
img3_lvl = display_otsu_level(img3_color,tau,l,x,y,i)
dice3_lvl = dice(mask3, img3_lvl)

# Image 4
img4_color = cv2.imread("images_test/img4.jpg")
img4_lvl = display_otsu_level(img4_color,tau,l,x,y,i)
dice4_lvl = dice(mask4, img4_lvl)

# Image 5
img5_color = cv2.imread("images_test/img5.jpg")
img5_lvl = display_otsu_level(img5_color,tau,l,x,y,i)
dice5_lvl = dice(mask5, img5_lvl)

# Image 6
img6_color = cv2.imread("images_test/img6.jpg")
img6_lvl = display_otsu_level(img6_color,tau,l,x,y,i)
dice6_lvl = dice(mask6, img6_lvl)

# Image 7
img7_color = cv2.imread("images_test/img7.jpg")
img7_lvl = display_otsu_level(img7_color,tau,l,x,y,i)
dice7_lvl = dice(mask7, img7_lvl)

# Image 8
img8_color = cv2.imread("images_test/img8.jpg")
img8_lvl = display_otsu_level(img8_color,tau,l,x,y,i)
dice8_lvl = dice(mask8, img8_lvl)

# Image 9
img9_color = cv2.imread("images_test/img9.jpg")
img9_lvl = display_otsu_level(img9_color,tau,l,x,y,i)
dice9_lvl = dice(mask9, img9_lvl)

# Image 10
img10_color = cv2.imread("images_test/img10.jpg")
img10_lvl = display_otsu_level(img10_color,tau,l,x,y,i)
dice10_lvl = dice(mask10, img10_lvl)

# Image 11
img11_color = cv2.imread("images_test/img11.jpg")
img11_lvl = display_otsu_level(img11_color,tau,l,x,y,i)
dice11_lvl = dice(mask11, img11_lvl)

# Image 12
img12_color = cv2.imread("images_test/img12.jpg")
img12_lvl = display_otsu_level(img12_color,tau,l,x,y,i)
dice12_lvl = dice(mask12, img12_lvl)

# Image 13
img13_color = cv2.imread("images_test/img13.jpg")
img13_lvl = display_otsu_level(img13_color,tau,l,x,y,i)
dice13_lvl = dice(mask13, img13_lvl)

# Image 14
img14_color = cv2.imread("images_test/img14.jpg")
img14_lvl = display_otsu_level(img14_color,tau,l,x,y,i)
dice14_lvl = dice(mask14, img14_lvl)

# Image 15
img15_color = cv2.imread("images_test/img15.jpg")
img15_lvl = display_otsu_level(img15_color,tau,l,x,y,i)
dice15_lvl = dice(mask15, img15_lvl)

# Image 16
img16_color = cv2.imread("images_test/img16.jpg")
img16_lvl = display_otsu_level(img16_color,tau,l,x,y,i)
dice16_lvl = dice(mask16, img16_lvl)

# Image 17
img17_color = cv2.imread("images_test/img17.jpg")
img17_lvl = display_otsu_level(img17_color,tau,l,x,y,i)
dice17_lvl = dice(mask17, img17_lvl)

# Image 18
img18_color = cv2.imread("images_test/img18.jpg")
img18_lvl = display_otsu_level(img18_color,tau,l,x,y,i)
dice18_lvl = dice(mask18, img18_lvl)

# Image 19
img19_color = cv2.imread("images_test/img19.jpg")
img19_lvl = display_otsu_level(img19_color,tau,l,x,y,i)
dice19_lvl = dice(mask19, img19_lvl)

# Image 20
img20_color = cv2.imread("images_test/img20.jpg")
img20_lvl = display_otsu_level(img20_color,tau,l,x,y,i)
dice20_lvl = dice(mask20, img20_lvl)

D_lvl = [[dice1_lvl], [dice2_lvl], [dice3_lvl], [dice4_lvl], [dice5_lvl], [dice6_lvl], [dice7_lvl], [dice8_lvl], [dice9_lvl], [dice10_lvl], [dice11_lvl], [dice12_lvl], [dice13_lvl], [dice14_lvl], [dice15_lvl], [dice16_lvl], [dice17_lvl], [dice18_lvl], [dice19_lvl], [dice20_lvl]]

def table_score (D):
    result1 = ["Image 1",D[0][0], D[0][1], D[0][2],D[0][3]]
    result2 = ["Image 2",D[1][0], D[1][1], D[1][2],D[1][3]]
    result3 = ["Image 3",D[2][0], D[2][1], D[2][2],D[2][3]]
    result4 = ["Image 4",D[3][0], D[3][1], D[3][2],D[3][3]]
    result5 = ["Image 5",D[4][0], D[4][1], D[4][2],D[4][3]]
    result6 = ["Image 6",D[5][0], D[5][1], D[5][2],D[5][3]]
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
    result20 = ["Image 20",D[19][0], D[19][1], D[19][2],D[19][3]]
    results = [result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20]



    table = "| Image | Otsu | Otsu + Pre-processing 1 | Otsu + Post-processing 1 | Otsu + Pre-processing 1 + Post-processing 1 \n"
    table += "| --- | --- | --- | --- | --- \n"
    for result in results:
        table += "| " + " | ".join([str(r) for r in result]) + " |\n"

    display(Markdown(table))

    def table_score_lvl(D):
        result1 = ["Image 1",D[0][0]]
        result2 = ["Image 2",D[1][0]]
        result3 = ["Image 3",D[2][0]]
        result4 = ["Image 4",D[3][0]]
        result5 = ["Image 5",D[4][0]]
        result6 = ["Image 6",D[5][0]]
        result7 = ["Image 7",D[6][0]]
        result8 = ["Image 8",D[7][0]]
        result9 = ["Image 9",D[8][0]]
        result10 = ["Image 10",D[9][0]]
        result11 = ["Image 11",D[10][0]]
        result12 = ["Image 12",D[11][0]]
        result13 = ["Image 13",D[12][0]]
        result14 = ["Image 14",D[13][0]]
        result15 = ["Image 15",D[14][0]]
        result16 = ["Image 16",D[15][0]]
        result17 = ["Image 17",D[16][0]]
        result18 = ["Image 18",D[17][0]]
        result19 = ["Image 19",D[18][0]]
        result20 = ["Image 20",D[19][0]]
        results = [result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20]

        table = "| Image | Otsu_level \n"
        table += "| --- | --- \n"
        for result in results:
            table += "| " + " | ".join([str(r) for r in result]) + " |\n"

        display(Markdown(table))