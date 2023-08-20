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


def draw_lines(image, lines, color=[255, 0, 0], thickness=3):
    """       
    Lines are drawn on the image with params 'color' and 'thickness' inplace.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, thresh, min_line_len, max_line_gap):
    """        
    Returns an image with hough lines drawn.
    """
    print(f"Drawing hough lines...")
    lines = cv2.HoughLinesP(image, rho, theta, thresh, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    draw_lines(line_image,lines)
    return line_image

def weighted_img(line_image, initial_image, vertices, alpha=0.1, beta=1., gamma=0):
    """    
    The result image is computed as follows:
    
    initial_image * alpha + image * beta + gamma (initial_image and image should be the same shape)
    """
    print(f"Applying weighted image...")
    lines_edges = cv2.addWeighted(initial_image, alpha, line_image, beta, gamma)
    lines_edges = cv2.polylines(lines_edges,vertices, True, (0,0,255), 1)
    return lines_edges

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.15, rows]
    top_left     = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver