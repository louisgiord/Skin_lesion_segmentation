# dull razo reprise

#### PROBLEME A LA LIGNE 98 ####
#%% bibliothèques
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage 
import skimage.io as skio
import skimage.morphology as morpho
import skimage.segmentation as seg
from skimage import color
from skimage.restoration import inpaint

#%% Charger l'image - changer le path sur le git
image_rgb = skio.imread('ISIC_0000042.jpg')
#%% application filtre médian puis split des couleurs 
image_red, image_green, image_blue = cv2.split(image_rgb)
selem = np.ones((6,6))
image_med_r = skimage.filters.median(image_red, footprint= selem)
image_med_g = skimage.filters.median(image_green, footprint = selem)
image_med_b = skimage.filters.median(image_blue, footprint = selem)
plt.imshow(image_red, cmap='gray')
plt.show()
plt.imshow(image_med_r, cmap='gray')
plt.show()


#%% Définir les éléments structurants - on peut changer les valeurs pour avoir un meilleur résultat

#%% NE PAS RUN
# définition d'éléments structurants diagonaux 
def diagonal(size, angle):
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(i * np.tan(np.radians(angle)))
        if 0<= j < size:
            kernel[i, j] = 1
    if angle > 0 and angle < 90:
        kernel = np.flipud(kernel)
    return kernel

# Exemple d'utilisation
kernel_diagonal_30 = diagonal(10,30)
print(kernel_diagonal_30)
#print(diagonal(10, 135))

#%% kernel vertical et horizontal
kernel_vertical = morpho.rectangle(15, 1)
print(kernel_vertical)
kernel_horizontal = morpho.rectangle(1, 15)
print(kernel_horizontal)

#%% Appliquer une opération de fermeture morphologique générale à chaque bande de couleur
def morphological_opening(image, kernel):
    return morpho.opening(image, kernel)
#rouge
clos_h_r = morphological_opening(image_red, kernel_horizontal)
clos_45_r = morphological_opening(image_red, diagonal(15,45))
clos_v_r = morphological_opening(image_red, kernel_vertical)
#vert
clos_h_g = morphological_opening(image_green, kernel_horizontal)
clos_45_g = morphological_opening(image_green, diagonal(15,45))
clos_v_g = morphological_opening(image_green, kernel_vertical)
#bleu
clos_h_b = morphological_opening(image_blue, kernel_horizontal)
clos_45_b = morphological_opening(image_blue, diagonal(15,45))
clos_v_b = morphological_opening(image_blue, kernel_vertical)

#%% calcul Gr : generalized graysacle closing
g_r = np.abs(image_red - np.max([clos_h_r, clos_45_r, clos_v_r]))
plt.imshow(g_r, cmap='gray')
plt.show()

g_g = np.abs(image_green - np.max([clos_h_g, clos_45_g, clos_v_g]))
plt.imshow(g_g, cmap='gray')
plt.show()

g_b = np.abs(image_blue - np.max([clos_h_b, clos_45_b, clos_v_b]))
plt.imshow(g_b, cmap='gray')
plt.show()

#%%resize
g_r = cv2.resize(g_r, (image_rgb.shape[1], image_rgb.shape[0]))
g_g = cv2.resize(g_g, (image_rgb.shape[1], image_rgb.shape[0]))
g_b = cv2.resize(g_b, (image_rgb.shape[1], image_rgb.shape[0]))

#%% calcul mask hair
hist, bins = np.histogram(g_r.ravel(), 256, [0, 256])
threshold = np.argmax(hist)

mask_r = np.where(g_r > threshold, 1, 0)
mask_g = np.where(g_g > threshold, 1, 0)
mask_b = np.where(g_b > threshold, 1, 0)

plt.imshow(mask_r, cmap='gray')
plt.show()

plt.imshow(mask_g, cmap='gray')
plt.show()

plt.imshow(mask_b, cmap='gray')
plt.show()

mask = np.logical_or(np.logical_or(mask_r, mask_g), mask_b)
plt.imshow(mask, cmap='gray')
plt.show()
#%% Remplacer les pixels de cheveux par les pixels non-cheveux les plus proches
inpainted_image = inpaint.inpaint_biharmonic(image_rgb, mask)
#%% affichage
plt.imshow(inpainted_image)
plt.show()





#%% Calculer l'image de fermeture morphologique générale pour chaque bande de couleur
"""def general_morphological_closing(image):
    closed_horizontal = morphological_closing(image, kernel_horizontal)
    closed_diagonal = morphological_closing(image, diagonal(15, 30))
    closed_vertical = morphological_closing(image, kernel_vertical)
    return np.maximum(np.maximum(closed_horizontal, closed_diagonal), closed_vertical)
"""
def morphologie(image_col, kernel):
    #closed = morpho.closing(image_rgb, kernel)
    image_rdg = skimage.filters.median(image_col)
    image_red, image_green, image_blue = cv2.split(image_rgb)
    closed_red = morphological_closing(image_red, kernel)
    closed_green = morphological_closing(image_green, kernel)
    closed_blue = morphological_closing(image_blue, kernel)
    # Calculer l'image masque de cheveux pour chaque bande de couleur
    threshold = 30  # Définir un seuil prédéfini - peut être ajusté car gros poils persistants
    hair_mask_red = np.abs(image_red - closed_red) > threshold
    hair_mask_green = np.abs(image_green - closed_green) > threshold
    hair_mask_blue = np.abs(image_blue - closed_blue) > threshold
    hair_mask = np.logical_or(np.logical_or(hair_mask_red, hair_mask_green), hair_mask_blue).astype(np.uint8)
    #règler le problèem de taille
    hair_mask = cv2.resize(hair_mask, (image_rgb.shape[1], image_rgb.shape[0]))
    print(image_rgb.shape)
    print(hair_mask.shape)
    return hair_mask

mask_horizontal = morphologie(image_rgb, kernel_horizontal)
plt.imshow(mask_horizontal, cmap='gray')
plt.show()
mask_vertical = morphologie(image_rgb, kernel_vertical)
#plt.imshow(mask_vertical, cmap='gray')
#plt.show()
mask_diagonal_30 = morphologie(image_rgb, diagonal(15, 30))
mask_diagonal_45 = morphologie(image_rgb, diagonal(15, 45))
plt.imshow(mask_diagonal_45, cmap='gray')
plt.show()
mask_diagonal_60 = morphologie(image_rgb, diagonal(15, 60))
mask_diagonal_75 = morphologie(image_rgb, diagonal(15, 75))
mask_diagonal_105 = morphologie(image_rgb, diagonal(15, 105))
mask_diagonal_120 = morphologie(image_rgb, diagonal(15, 120))
mask_diagonal_135 = morphologie(image_rgb, diagonal(15, 135))

#on prend le minimum car le noir est 
mask = np.maximum(mask_horizontal, mask_vertical)
mask = np.maximum(mask, mask_diagonal_30)
mask = np.maximum(mask, mask_diagonal_45)
mask = np.maximum(mask, mask_diagonal_60)
plt.imshow(mask, cmap='gray')
plt.show()
mask = np.maximum(mask, mask_diagonal_75)
mask = np.maximum(mask, mask_diagonal_105)
mask = np.maximum(mask, mask_diagonal_120)
plt.imshow(mask, cmap='gray')
plt.show()
mask = np.maximum(mask, mask_diagonal_135)
mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

#%% Remplacer les pixels de cheveux par les pixels non-cheveux les plus proches
#hair_mask = morphologie(image_rgb)
inpainted_image = inpaint.inpaint_biharmonic(image_rgb, mask)
#%% Appliquer un lissage à l'image finale
#smoothed_image = cv2.GaussianBlur(inpainted_image, (5, 5), 0)


#%% en pyplot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(222)
plt.imshow(mask * 255, cmap='gray')
plt.title('Hair Mask')

plt.subplot(223)
plt.imshow(inpainted_image)
plt.title('Inpainted Image')

#plt.subplot(224)
#plt.imshow(smoothed_image)
#plt.title('Smoothed Image')

plt.show()
