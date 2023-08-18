import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform and returns grayscale image"""
    print(f"Applying Grayscale...")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

