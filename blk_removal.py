import numpy as np 
import  cv2
import matplotlib
from otsu_seg import viewimage

#parameters

l = 5 # width of the set 5 pixels
S=[0,l]

# Load the image
img=cv2.imread ("images_test/im_test3.jpg")

#Transform the image to gray scale
img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape[0]/float(4),img.shape[1]/float(4))

#Means of a set of pixel

def means_set(S,img,x,y):
    m=0
    N=S[0]*S[1]
    for i in range (x,x+1+S[0]):
        for j in range (y,y+1+S[1]):
            m+= img[j,i]
    return m/N

def remove_column (img,lenght,S,x,y):
    Ly = img.shape[0]
    Lx = img.shape[1]
    S[0]= lenght
    ratio = 1
    mean_optimal = 0
    i_optimal = 0   
    while ratio > 0.5:
        for i in range(0,int(Lx/2),S[1]):
            black_column = []
            mean_column = []
            for j in range (0,int(Ly),lenght):
                mean_local = means_set(S,img,x,j)
                if mean_local < 25 :
                    black_column.append(1)
                mean_column.append(mean_local)
            s = np.sum(black_column)
            ratio = s/(Ly/float(lenght))
            mean_optimal = np.mean(mean_column)
            i_optimal = i
    for i in range(i_optimal*S[1]):
            for j in range(0,Ly):
                img[j,i] = mean_optimal
    return img  
        

viewimage(remove_column(img,4,S,0,0))




