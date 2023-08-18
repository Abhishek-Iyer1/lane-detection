import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform and returns grayscale image"""
    print(f"Applying Grayscale...")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, kernel_size):
    """Applies a Gaussian Noise kernel"""
    print(f"Applying Gaussian Blur...")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_thresh, high_thresh):
    """Applies the Canny transform"""
    print(f"Applying Canny Edge Detection...")
    return cv2.Canny(image, low_thresh, high_thresh)

def apply_mask(image, vertices):
    """
    Applies and returns image mask defined by 'vertices'.

    Pixels outside the vertices are set to black. 
    """
    print(f"Masking region of interest...")
    # Defining mask color based on n_channels in image
    if len(image.shape) > 2:
        n_channels = image.shape[2]
        mask_color = (255,) * n_channels
    else:
        mask_color = 255

    # Initializing blank mask
    mask = np.zeros_like(image)   
        
    # Fills pixels within the vertices with given color    
    cv2.fillPoly(mask, vertices, mask_color)
    
    # Applies mask on the image in a bitwise 'and' operation
    image_with_mask = cv2.bitwise_and(image, mask)
    return image_with_mask


