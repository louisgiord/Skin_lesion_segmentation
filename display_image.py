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

def viewimage(img):
    if img is None:
        print("Erreur : l'image n'a pas été chargée correctement.")
        return
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
