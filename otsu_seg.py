#implementing otsu segmentation
#the goal of the algorithme is to distinghish the pixels of the image into two classes C0 and C1. 


import cv2
import numpy as np
from matplotlib import pyplot as plt
from blk_removal import mask_remove
from display_image import viewimgs,viewimage

#parameters
N = 256 #number of gray levels
tau = 60 
x,y = 20,20
l=5


def plot_norm_hist(img):
    hist = cv2.calcHist([img], [0], None, [N], [0, N])
    plt.plot(hist, color='gray' )
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

def prob_gray_lvl(img,mask): 
    hist = cv2.calcHist([img], [0], mask, [N], [0, N]) #histogram
    num_pixels = np.sum(hist)
    proba_level = hist/num_pixels
    return proba_level

#compute the probability of occurence C0, going to the gray level of value t, and of C1
def proba_class_thresh(img,mask, t): 
    proba_level = prob_gray_lvl(img,mask)
    #threshold is the gray level that separates the two classes
    p,q = np.sum(proba_level[:t]), np.sum(proba_level[t:])
    #print(f"p: {p}, q: {q}")
    #WARNING !!!! sometines value zero
    return p,q

def mean_class_thresh(img,mask,t):
    mean = 0
    proba_level = prob_gray_lvl(img,mask)
    for i in range (1,t+1):
        mean += i*proba_level[i-1]
    return mean

def mean_C0(img,mask,t):
    w= proba_class_thresh(img,mask,t)[0]
    mu= mean_class_thresh(img,mask,t)
    return mu/w

def mean_C1(img,mask,t):
    mu = mean_class_thresh(img,mask,N) - mean_class_thresh(img,mask,t)
    w = proba_class_thresh(img,mask,t)[1]
    return mu/w

def var_between_class(img,mask,t):
    w0,w1 = proba_class_thresh(img,mask,t)
    mu0 = mean_C0(img,mask,t)
    mu1 = mean_C1(img,mask,t)
    return w0*w1*(mu0-mu1)**2

#the final goal of the algorithme is to find the threshold that maximizes the between class variance

def otsu(img,mask):
    var_max = 0
    tresh = 0
    for i in range(N):
        var = var_between_class(img,mask,i)
        if var > var_max:
            var_max = var
            tresh = i
    return tresh

img = cv2.imread('images_test/img1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = mask_remove(img,tau,l,x,y)
tresh = otsu(img,mask)
tresh1 = otsu(img,None)
print(f"threshold: {tresh}")
print(f"threshold1: {tresh1}")
ret,mask1 = cv2.threshold(img,tresh,255,cv2.THRESH_BINARY)
ret,mask2 = cv2.threshold(img,tresh1,255,cv2.THRESH_BINARY)
viewimgs(mask1,mask2)

