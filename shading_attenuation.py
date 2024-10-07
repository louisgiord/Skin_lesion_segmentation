import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from otsu_seg import viewimages

# Load the colored image 

img_rgb = cv2.imread("images_test/im_test1.jpg")
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

# Extract Hue, Value and Saturation

hue, sat, val = cv2.split(img_hsv)

#parameters
x, y = 50, 50
w, h = 50, 50
param = [0,0,0,0,0,0]

#Extract the submask
def extract_submask(x,y,w,h): 
    val_sub = val[y:y+h, x:x+h] #In matrix convention with lines and columns
    return val_sub

def z(x,y,param):
    return param[0]*(x**2) + param[1]*(y**2) + param[2]*x*y + param[3]*x + param[4]*y + param[5]

def error(param,w,h):
    E = 0 
    submask = extract_submask(x,y,w,h)
    for i in range (w):
        for j in range (h):
            E += (submask[j,i] - z(i,j,param))**2
    return E

result = minimize(error, param, args=(w,h), method='Nelder-Mead')
print (result.x)
    
def new_val_pixel(x,y,param):
    M = z(x,y,param)
    return val[y,x]/M

def new_val_image(param):
    new_val = np.zeros(val.shape)
    for i in range (val.shape[1]):
        for j in range (val.shape[0]):
            new_val[j,i] = new_val_pixel(i,j,param)
    return new_val

val = new_val_image(result.x)

new_hsv_image = cv2.merge([hue,sat,val])
new_rgb_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB)

viewimages(img_rgb, new_hsv_image)
