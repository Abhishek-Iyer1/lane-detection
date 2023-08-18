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

