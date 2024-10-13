import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from otsu_seg import view2images, viewimage

# Load the colored image 

img_rgb = cv2.imread("images_test/im_test1.jpg")
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

#view2images(img_rgb, img_hsv) #can de siplayed in the same window because they got the same number of canals

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
print (result.x) #print the optimized parameters which minimize the error

def val_tild_pix(x,y,param):
    R = z(x,y,param)
    I = val[y,x]
    return I/R

def val_tilde(param):
    new_val = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=img_rgb.dtype)#image with the same size as the original image
                                                           #with only zeros
    for i in range(img_rgb.shape[1]):
        for j in range(img_rgb.shape[0]):
            new_val[j,i] = val_tild_pix(i,j,param)
    return new_val



#new_val = val_tilde(result.x)    

#im_tilde_hsv = np.dstack((hue, sat, new_valval))
im_tilde_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

#view2images(img_hsv, im_tilde_hsv)
view2images(img_rgb, im_tilde_rgb)





    
