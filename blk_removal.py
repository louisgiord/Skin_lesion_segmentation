import numpy as np 
import  cv2
import matplotlib
from otsu_seg import viewimage, view2images

# Load the image
img=cv2.imread ("images_test/img1.jpg")

#Check is the image is correctly charged
if img is None:
    raise FileNotFoundError("image not found")

#Transform the image to gray scale
img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#test if the columns are only black pixels

def test_black_column(img,blk_tresh):
    Ly = img.shape[0] #number of lines
    Lx = img.shape[1] #number of columns
    column_index =[]
    i = 0
    m = 1
    while i < Lx and m == 1 :
        test = []
        for j in range(Ly):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            column_index.append(i)
        i += 1   
    return column_index

def test_black_column_rverse(img,blk_tresh):
    Ly=img.shape[0] #number of lines
    Lx=img.shape[1] #number of columns
    column_index=[]
    i = Lx - 1
    m = 1 
    while i > 0 and m == 1 : 
        test = []
        for j in range (Ly):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            column_index.append(i)
        i -= 1
    return column_index


def blk_column_index(img,blk_tresh):
    return test_black_column(img,blk_tresh) + test_black_column_rverse(img,blk_tresh)

def test_black_line(img,blk_tresh):
    Ly = img.shape[0]
    Lx = img.shape[1]
    line_index = []
    j = 0 
    m = 1
    while j < Ly and m == 1:
        test = []
        for i in range (Lx):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            line_index.append(j)
        j += 1
    return line_index

def test_black_line_rverse(img,blk_tresh):
    Ly = img.shape[0]
    Lx = img.shape[1]
    line_index = []
    j = Ly - 1
    m = 1
    while j > 0 and m == 1:
        test = []
        for i in range (Lx):
            if img[j,i] < blk_tresh:
                test.append(1)
            else: 
                test.append(0)
        m = np.mean(test)
        if m == 1:
            line_index.append(j)
        j -= 1
    return line_index

def blk_line_index(img,blk_tresh):
    return test_black_line(img,blk_tresh)+test_black_line_rverse(img, blk_tresh)

#parameters

T = 30

#viewimage(img_gray)
#viewimage(img)
#view2images(img, img_gray) pb of size between the two images

# generate an image with a black column in the middle and elsewhere white pixels
im_black = np.zeros(img_gray.shape)
im_black[:,:] = 255
im_black[:600,] = 0
viewimage(im_black)

index_col = blk_column_index(im_black, T)
index_line = blk_line_index(im_black, T)
print("index of lines full of black",index_col)
print("index of columns full of black",index_line)

im_modif = np.delete(im_black,index_line, axis = 0)
viewimage(im_modif)

#im_black = np.zeros(img_gray.shape)
#viewimage(im_black)










