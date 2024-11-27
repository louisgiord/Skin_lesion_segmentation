import cv2
import numpy as np
import cv2
import numpy as np


def find_largest_connected_component(binary_img):
    # Étiquetage des composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    # Trouver la composante connexe avec la plus grande aire (ignorer le label 0 qui est le fond)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Créer une image binaire pour la composante connexe principale
    largest_component = np.zeros_like(binary_img, dtype=np.uint8)
    largest_component[labels == largest_label] = 255
    return largest_component