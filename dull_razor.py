# dull razor : hair remover
import cv2 #choix de openCv (on aurait pu faire avec skimage)
import numpy as np
import matplotlib.pyplot as plt

#Charger l'image - changer le path sur le git

image_bgr=cv2.imread('img19.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Conversion BGR vers RGB
#filtre gaussien pour lisser l'image 
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)


# split en couleurs 
image_red, image_green, image_blue = cv2.split(smoothed_image)

# Définir les éléments structurants - on peut changer les valeurs pour avoir un meilleur résultat
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
kernel_diagonal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

# Appliquer une opération de fermeture morphologique générale à chaque bande de couleur
def morphological_closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Calculer l'image de fermeture morphologique générale pour chaque bande de couleur
def general_morphological_closing(image):
    closed_horizontal = morphological_closing(image, kernel_horizontal)
    closed_diagonal = morphological_closing(image, kernel_diagonal)
    closed_vertical = morphological_closing(image, kernel_vertical)
    
    closed_horizontal = cv2.resize(closed_horizontal, (image.shape[1], image.shape[0]))
    closed_diagonal = cv2.resize(closed_diagonal, (image.shape[1], image.shape[0]))
    closed_vertical = cv2.resize(closed_vertical, (image.shape[1], image.shape[0]))

    return np.maximum(np.maximum(closed_horizontal, closed_diagonal), closed_vertical)

closed_red = general_morphological_closing(image_red)
closed_green = general_morphological_closing(image_green)
closed_blue = general_morphological_closing(image_blue)

# Calculer l'image masque de cheveux pour chaque bande de couleur
threshold = 75  # Définir un seuil prédéfini - peut être ajusté car gros poils persistants
hair_mask_red = np.abs(image_red - closed_red) > threshold
hair_mask_green = np.abs(image_green - closed_green) > threshold
hair_mask_blue = np.abs(image_blue - closed_blue) > threshold

# Calculer l'image masque de cheveux finale
hair_mask = np.logical_or(np.logical_or(hair_mask_red, hair_mask_green), hair_mask_blue).astype(np.uint8)

# Remplacer les pixels de cheveux par les pixels non-cheveux les plus proches sur l'image originale
inpainted_image = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(222)
plt.imshow(hair_mask * 255, cmap='gray')
plt.title('Hair Mask')

plt.subplot(223)
plt.imshow(inpainted_image)
plt.title('Inpainted Image')

plt.subplot(224)
plt.imshow(smoothed_image)
plt.title('Smoothed Image initiale')

plt.show()

