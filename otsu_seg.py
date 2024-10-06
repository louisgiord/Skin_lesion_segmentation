#implementing otsu segmentation
#the goal of the algorithme is to distinghish the pixels of the image into two classes C0 and C1. 

import cv2
import numpy as np
from matplotlib import pyplot as plt

#parameters
N = 256 #number of gray levels

# Load the image
img=cv2.imread ("im_test1.jpg")

#Transform the image to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#display images 
def viewimage(img):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_norm_hist(img):
    hist = cv2.calcHist([img], [0], None, [N], [0, N])
    plt.plot(hist, color='gray' )
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

def prob_gray_lvl(img):
    data_level = np.zeros(N) #array with number of pixel corresponding to each gray level
    num_pixels =  img.shape[0]*img.shape[1] #total number of pixels
    hist = cv2.calcHist([img], [0], None, [N], [0, N]) #histogram
    for i in range (N):
        data_level[i] = hist[i][0]
    proba_level = data_level/num_pixels
    return proba_level

#compute the probability of occurence C0, going to the gray level of value t, and of C1
def proba_class_thresh(img, t): 
    proba_level = prob_gray_lvl(img)
    #threshold is the gray level that separates the two classes
    p,q = np.sum(proba_level[:t]), np.sum(proba_level[t:])
    #print(f"p: {p}, q: {q}")
    #WARNING !!!! sometines value zero
    return p,q

def mean_class_thresh(img,t):
    mean = 0
    proba_level = prob_gray_lvl(img)
    for i in range (1,t+1):
        mean += i*proba_level[i-1]
    return mean

def mean_C0(img,t):
    w= proba_class_thresh(img,t)[0]
    mu= mean_class_thresh(img,t)
    return mu/w

def mean_C1(img,t):
    mu = mean_class_thresh(img,N) - mean_class_thresh(img,t)
    w = proba_class_thresh(img,t)[1]
    return mu/w

def var_between_class(img,t):
    w0,w1 = proba_class_thresh(img,t)
    mu0 = mean_C0(img,t)
    mu1 = mean_C1(img,t)
    return w0*w1*(mu0-mu1)**2

#the final goal ogf the algorithme is to find the threshold that maximizes the between class variance

def otsu(img):
    var_max = 0
    tresh = 0
    for i in range(N):
        var = var_between_class(img,i)
        if var > var_max:
            var_max = var
            tresh = i
    return tresh

#test the algorithm
tresh = otsu(img_gray)
print(f"Optimal threshold: {tresh}")

#apply the threshold to the image
ret,thresh1 = cv2.threshold(img_gray,tresh,255,cv2.THRESH_BINARY)
viewimage(thresh1)
