import numpy as np 
import cv2
import matplotlib.pyplot as plt
import scipy.optimize as opt
from display_image import viewimgs, viewimage

# Load the colored image 

img_rgb = cv2.imread("images_test/img2.jpg")
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

#viewimage(img_hsv)

# Extract Hue, Value and Saturation

hue, sat, val = cv2.split(img_hsv)

#Extract the submask
def extract_submask(x,y,l,img_hsv): 
    hue, sat, val = cv2.split(img_hsv)
    val_sub = val[y:y+l,x:x+l] #In matrix convention with lines and columns
    m = np.mean(val_sub)
    return val_sub,m

def z(x,y,param):
    return param[0]*(x**2) + param[1]*(y**2) + param[2]*x*y + param[3]*x + param[4]*y + param[5]

def error(param,img_hsv,l,x,y):
    hue, sat, val = cv2.split(img_hsv)
    for i in range(l):
        for j in range(l):
            Val = val[y+j,x+i]
            z_val = z(x+i,y+j,param)
    e = np.sum((Val - z_val)**2)
    return e

def R(x,y,param,img_hsv):
    hue, sat, val = cv2.split(img_hsv)
    V = val[y,x]
    Z_val = z(x,y,param)
    return V/Z_val

def newVal(img_hsv,param):
    Lx = img_hsv.shape[1]
    Ly = img_hsv.shape[0]
    new_val = np.zeros((Ly,Lx))
    for i in range(Lx):
        for j in range(Ly):
            new_val[j,i] = R(i,j,param,img_hsv)
    return new_val

def shading_attenuation(img_hsv,l,x,y):
    ini_param = np.random.rand(6)
    result = opt.least_squares(error,ini_param,args=(img_hsv,l,x,y))
    opt_param = result.x
    min_error = result.cost
    val_new = newVal(img_hsv,opt_param)
    new_img_hsv = img_hsv.copy()
    new_img_hsv[:,:,2] = val_new
    new_img_rgb = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2RGB)
    return new_img_rgb


#Parameters 
l = 100
x = 150
y = 150
ini_param = np.random.rand(6)

viewimage(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
new_img_rgb = cv2.cvtColor(shading_attenuation(img_hsv,l,x,y), cv2.COLOR_HSV2GBR)