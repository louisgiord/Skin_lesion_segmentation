import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def viewimgs(img1, img2): #WARNING: img1 and img2 must have the same number of canals
    # convert to RGB for matplotlib, not the same convention as OpenCV
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
    
    # Resize img2 if different size
    if img1.shape != img2.shape:
        img2_rgb = cv2.resize(img2_rgb, (img1.shape[1], img1.shape[0]))
    
    # concatenate the two images
    img_conc = np.hstack((img1_rgb, img2_rgb))
    
    # Plot the images
    plt.imshow(img_conc, cmap='gray' if len(img_conc.shape) == 2 else None)
    plt.axis('off')  # Masquer les axes
    plt.show()


def view3imgs(img1, img2, img3):  # WARNING: img1, img2, and img3 must have the same number of channels
    # Convert to RGB for matplotlib, not the same convention as OpenCV
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) if len(img3.shape) == 3 else img3
    
    # Resize img2 and img3 if different size
    if img1.shape != img2.shape:
        img2_rgb = cv2.resize(img2_rgb, (img1.shape[1], img1.shape[0]))
    if img1.shape != img3.shape:
        img3_rgb = cv2.resize(img3_rgb, (img1.shape[1], img1.shape[0]))
    
    # Concatenate the three images
    img_conc = np.hstack((img1_rgb, img2_rgb, img3_rgb))
    
    # Plot the images
    plt.imshow(img_conc, cmap='gray' if len(img_conc.shape) == 2 else None)
    plt.axis('off')  # Masquer les axes
    plt.show()


def viewimage(img):
    if img is None:
        print("Erreur : l'image n'a pas été chargée correctement.")
        return
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mask_display(img,mask,tresh):
    result = cv2.threshold(img,tresh,255,cv2.THRESH_BINARY_INV)[1]
    if mask is None:
        return result
    else:
        for i in range (img.shape[0]):
            for j in range (img.shape[1]):
                if mask[i,j] == 0:
                    result[i,j] = 0
    return result

