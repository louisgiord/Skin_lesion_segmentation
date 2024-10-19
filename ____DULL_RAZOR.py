#%% bibliothèques
import cv2
import numpy as np

#%% Charger l'image - changer le path sur le git

image_bgr=cv2.imread('skin_lesion_dataset/melanoma/ISIC_0000146.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Conversion BGR vers RGB
#%% split en couleurs 
image_red, image_green, image_blue = cv2.split(image)

#%% Définir les éléments structurants
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
kernel_diagonal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

#%% Appliquer une opération de fermeture morphologique générale à chaque bande de couleur
def morphological_closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

#%% Calculer l'image de fermeture morphologique générale pour chaque bande de couleur
def general_morphological_closing(image):
    closed_horizontal = morphological_closing(image, kernel_horizontal)
    closed_diagonal = morphological_closing(image, kernel_diagonal)
    closed_vertical = morphological_closing(image, kernel_vertical)
    return np.maximum(np.maximum(closed_horizontal, closed_diagonal), closed_vertical)

closed_red = general_morphological_closing(image_red)
closed_green = general_morphological_closing(image_green)
closed_blue = general_morphological_closing(image_blue)

#%% Calculer l'image masque de cheveux pour chaque bande de couleur
threshold = 30  # Définir un seuil prédéfini - peut être ajusté car gros poils persistants
hair_mask_red = np.abs(image_red - closed_red) > threshold
hair_mask_green = np.abs(image_green - closed_green) > threshold
hair_mask_blue = np.abs(image_blue - closed_blue) > threshold

#%% Calculer l'image masque de cheveux finale
hair_mask = np.logical_or(np.logical_or(hair_mask_red, hair_mask_green), hair_mask_blue).astype(np.uint8)

#%% Remplacer les pixels de cheveux par les pixels non-cheveux les plus proches
inpainted_image = cv2.inpaint(image_bgr, hair_mask, 3, cv2.INPAINT_TELEA)

#%% Appliquer un lissage à l'image finale
smoothed_image = cv2.GaussianBlur(inpainted_image, (5, 5), 0)

#%% Afficher les résultats - mauvaise affichage sur mon ordinateur, essayer avec matplotlib
cv2.imshow('Original Image', image_bgr)
cv2.imshow('Hair Mask', hair_mask * 255)
cv2.imshow('Inpainted Image', inpainted_image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
