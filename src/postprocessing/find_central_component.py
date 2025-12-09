# Function to find the central component

import cv2
import numpy as np
import cv2
import numpy as np


def find_largest_connected_component(binary_img):
    # Labeling of connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    # Find the connected component with the largest area (ignore label 0 which is the background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a binary image for the main connected component
    largest_component = np.zeros_like(binary_img, dtype=np.uint8)
    largest_component[labels == largest_label] = 255
    return largest_component