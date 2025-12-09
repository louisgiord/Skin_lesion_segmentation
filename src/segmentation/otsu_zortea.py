#%% libraries
import cv2
import numpy as np
import skimage
from skimage import color
from matplotlib import pyplot as plt

# Parameters
N = 256  # Number of gray levels

#%% Load the image
img = cv2.imread("data/images_test/img1.jpg")

# Transform the image to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display 2 images with the same number of canals for comparison
def view2images(img1, img2):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img_conc = np.hstack((img1, img2))
    cv2.imshow('Comparison of images', img_conc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display a single image
def viewimage(img):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_norm_hist(img):
    hist = cv2.calcHist([img], [0], None, [N], [0, N])
    plt.plot(hist, color='gray')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

def prob_gray_lvl(img):
    data_level = np.zeros(N)  # Array with number of pixel corresponding to each gray level
    num_pixels = img.shape[0] * img.shape[1]  # Total number of pixels
    hist = cv2.calcHist([img], [0], None, [N], [0, N])  # Histogram
    for i in range(N):
        data_level[i] = hist[i][0]
    proba_level = data_level / num_pixels
    return proba_level

# Compute the probability of occurrence C0, going to the gray level of value t, and of C1
def proba_class_thresh(img, t):
    proba_level = prob_gray_lvl(img)
    # Threshold is the gray level that separates the two classes
    p = np.sum(proba_level[:t])
    q = np.sum(proba_level[t:])
    return p, q

def mean_class_thresh(img, t):
    mean = 0
    proba_level = prob_gray_lvl(img)
    for i in range(1, t + 1):
        mean += i * proba_level[i - 1]
    return mean

def mean_C0(img, t):
    w = proba_class_thresh(img, t)[0]
    mu = mean_class_thresh(img, t)
    return mu / w if w != 0 else 0

def mean_C1(img, t):
    mu = mean_class_thresh(img, N) - mean_class_thresh(img, t)
    w = proba_class_thresh(img, t)[1]
    return mu / w if w != 0 else 0

def var_between_class(img, t):
    w0, w1 = proba_class_thresh(img, t)
    mu0 = mean_C0(img, t)
    mu1 = mean_C1(img, t)
    return w0 * w1 * (mu0 - mu1) ** 2

# The final goal of the algorithm is to find the threshold that maximizes the between class variance
def otsu(img):
    var_max = 0
    tresh = 0
    for i in range(N):
        var = var_between_class(img, i)
        if var > var_max:
            var_max = var
            tresh = i
    return tresh

#%% Beginning of Zortea
eta = 1/4
img = cv2.imread("img1.jpg")
L = max(img.shape[:2]) # Maximum dimension of the image
s = 0.02 * L # step size

# Convert the image to the CIELAB color space
img_lab = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Idea: move across the image with rectangles of size s, calculate standard deviation and mean
# up to eta of the distance from the smallest side of the image
# Select the skin region with minimal sum of standard deviation/mean ratio
def select_skin_region_init(img_gray):
    return img_gray[50:150, 50:150]
"""
def select_skin_region(img, s=0.02, eta=1/4):
    h, w = img.shape[:2]
    s_size = int(s * max(h, w))
    center_x, center_y = w // 2, h // 2
    min_ratio = float('inf')
    best_region = None
    
    for i in range(0, min(center_x, center_y), 5):
        x1, y1 = center_x - i, center_y - i
        x2, y2 = center_x + i, center_y + i
        region = img[y1:y2, x1:x2]
        mean = np.mean(region)
        ecart_type = np.std(region)
        ratio = ecart_type / mean
        if ratio < min_ratio:
            min_ratio = ratio
            best_region = region
    
    return best_region"""
def select_skin_region_bis(img_lab, s=0.02, eta=1/4):
    h, w = img_lab.shape[:2]
    s_size = int(s * max(h, w))
    center_img_x, center_img_y = w//2, h//2
    min_ratio = float('inf')
    best_region = None
# Iterate over rectangles of size s contained in the image borders at eta from the smallest side
    for i in range(0, h - s_size + 1, s_size):
        for j in range(0, w - s_size + 1, s_size):
            start_x = i
            end_x = i+s_size
            start_y = j
            end_y = j+s_size
            if end_x > h or end_y > w: # Verify that the region is contained within the image
                continue

            region = img_lab[start_x:end_x, start_y:end_y]
            mean_0 = np.mean(region[:,:,0])
            mean_1 = np.mean(region[:,:,1])
            mean_2 = np.mean(region[:,:,2])
            ecart_type_0 = np.std(region[:,:,0])
            ecart_type_1 = np.std(region[:,:,1])
            ecart_type_2 = np.std(region[:,:,2])
            ratio = sum([ecart_type_0/mean_0, ecart_type_1/mean_1, ecart_type_2/mean_2])
            if ratio < min_ratio:
                min_ratio = ratio
                best_region = region
    return best_region

# %% 2. Computation of an intensity image for thresholding
# Search for the median components of the pixels in the selected skin region
''' We choose the CIELAB because of its relative perceptual uniformity. 
Large (small) differences between any two colors correspond approximately 
to long (short) Euclidian distances between the colors in the three-dimensional CIELAB space.'''
Rs = select_skin_region_bis(img_lab, s, eta)
R_median = np.median(Rs, axis=(0, 1))

# Calculation of the intensity image
def image_intensity(img_lab, R_median, med):
    l, a, b = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    l_s, a_s, b_s = R_median
    intensity_image = np.sqrt((l - l_s) ** 2 + (a - a_s) ** 2 + (b - b_s) ** 2)
    ws = 0.01 * max(img.shape[:2])
    ws = int(ws)
    if med == 1:
        intensity_image_med = skimage.filters.median(intensity_image, footprint=np.ones((ws, ws)))
        return intensity_image_med
    return intensity_image

intensity_image = image_intensity(img_lab, R_median)

#%% 3.  Threshold estimation
# Select the cross_diagonal pixels with region size ws
def selection_2(image_lab):
    h, w = image_lab.shape[:2]
    ws = 0.01 * w
    ws = int(ws)
    center_x, center_y = w // 2, h // 2
    cross_diagonal = []
    i = 0
    for i in range(-h,h): # Traverse vertical pixels
        x, y = center_x, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w,w): # Traverse horizontal pixels
        x, y = center_x + i*ws, center_y
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w, w): # Traverse pixels from top-left to bottom-right diagonal
        x, y = center_x + i*ws, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w, w): # Traverse pixels from top-right to bottom-left diagonal
        x, y = center_x - i*ws, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    return cross_diagonal

def sigma(image,t):
    return var_between_class(image, t)


def phi(L_im, t):
    phi = []
    for i in range(len(L_im)):
        phi.append(sigma(L_im[i], t))
    # Calculation of the scaled Euclidean norm:
    phi = len(L_im)*np.linalg.norm(phi)
    return phi

# Calculate threshold_h first threshold
def threshold_h(image, R_median):
    th = sigma(image_intensity(image, R_median),0)
    for i in range(256):
        s1 = sigma(image_intensity(image, R_median), i)/phi([image_intensity(image, R_median), selection_2(img_lab)], i)
        s2 = sigma(selection_2(img_lab), i)/phi([image_intensity(image, R_median), selection_2(img_lab)], i)
        if s1 + s2 > th:    
            th = i
    return th

def threshold_s(img, beta):
    Rs = select_skin_region_bis(img, s, eta)
    nu1 = 0.05
    nu2 = 0.5
    im_i = image_intensity(Rs, R_median)
    gamma_1 = np.percentile(im_i, nu1)
    gamma_2 = np.percentile(im_i, nu2)
    return gamma_2 + beta*(gamma_2 - gamma_1)

def threshold_z(img, alpha, beta):
    if threshold_s(img, R_median) > threshold_h(img, beta):
        alpha = 1
    th_z = alpha*threshold_h(img, R_median) + (1-alpha)*threshold_s(img, beta)
    return th_z


# %% 4. Image segmentation
# Apply the final threshold to the intensity image
final_threshold = threshold_z(img, 0.5, 0.5)

img_seg = otsu(image_intensity(img_lab, R_median, 1))

# Display the segmented image
viewimage(img_seg)
