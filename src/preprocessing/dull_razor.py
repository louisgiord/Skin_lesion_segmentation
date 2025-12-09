#%% dull razor : hair remover
import cv2 # Choice of OpenCV (could have been done with skimage)
import numpy as np
import matplotlib.pyplot as plt

# Load the image - change the path on git

image_bgr=cv2.imread('data/images_test/img19.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR to RGB conversion
# Gaussian filter to smooth the image
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)


# Split into colors
image_red, image_green, image_blue = cv2.split(smoothed_image)

# Definition of diagonal structuring elements
def diagonal(size, angle):
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        j = int(i * np.tan(np.radians(angle)))
        if 0<= j < size:
            kernel[i, j] = 1
    if angle > 0 and angle < 90:
        kernel = np.flipud(kernel)
    return kernel

kernel_diagonal_45 = diagonal(10,45)
kernel_diagonal_135 = kernel_diagonal_45[::-1, :]

# Define the structuring elements - values can be changed for better results
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
kernel_diagonal_1 = kernel_diagonal_45
kernel_diagonal_2 = kernel_diagonal_135
kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))

# Apply a general morphological closing operation to each color band
def morphological_closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Compute the general morphological closing image for each color band
def general_morphological_closing(image):
    closed_horizontal = morphological_closing(image, kernel_horizontal)
    closed_diagonal_1 = morphological_closing(image, kernel_diagonal_45)
    closed_diagonal_2 = morphological_closing(image, kernel_diagonal_135)
    closed_vertical = morphological_closing(image, kernel_vertical)
    
    closed_horizontal = cv2.resize(closed_horizontal, (image.shape[1], image.shape[0]))
    closed_diagonal_1 = cv2.resize(closed_diagonal_1, (image.shape[1], image.shape[0]))
    closed_diagonal_2 = cv2.resize(closed_diagonal_2, (image.shape[1], image.shape[0]))
    closed_vertical = cv2.resize(closed_vertical, (image.shape[1], image.shape[0]))

    return np.maximum.reduce([closed_horizontal, closed_diagonal_1, closed_diagonal_2, closed_vertical])
closed_red = general_morphological_closing(image_red)
closed_green = general_morphological_closing(image_green)
closed_blue = general_morphological_closing(image_blue)

# Calculate the hair mask image for each color band
threshold = 30  # Define a predefined threshold - can be adjusted for persistent thick hairs
hair_mask_red = np.abs(image_red - closed_red) > threshold
hair_mask_green = np.abs(image_green - closed_green) > threshold
hair_mask_blue = np.abs(image_blue - closed_blue) > threshold

# Calculate the final hair mask image
hair_mask = np.logical_or(np.logical_or(hair_mask_red, hair_mask_green), hair_mask_blue).astype(np.uint8)

# Replace hair pixels with the nearest non-hair pixels on the original image
inpainted_image = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)

def dull_razor(image):
    image_bgr=cv2.imread(image,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR to RGB conversion
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    image_red, image_green, image_blue = cv2.split(smoothed_image)
    closed_red = general_morphological_closing(image_red)
    closed_green = general_morphological_closing(image_green)
    closed_blue = general_morphological_closing(image_blue)

# Calculate the hair mask image for each color band
    threshold = 30  # Define a predefined threshold - can be adjusted for persistent thick hairs
    hair_mask_red = np.abs(image_red - closed_red) > threshold
    hair_mask_green = np.abs(image_green - closed_green) > threshold
    hair_mask_blue = np.abs(image_blue - closed_blue) > threshold

# Calculate the final hair mask image
    hair_mask = np.logical_or(np.logical_or(hair_mask_red, hair_mask_green), hair_mask_blue).astype(np.uint8)

# Replace hair pixels with the nearest non-hair pixels on the original image
    inpainted_image = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
    inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
    return inpainted_image
'''
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
'''

# %%
