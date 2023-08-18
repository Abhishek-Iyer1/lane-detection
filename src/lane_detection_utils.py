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

def calculate_poly(image,lines):
    print(f"Calculating polygon and drawing lines...")
    image_copy = image.copy()
    poly_vertices = []
    order = [0,1,3,2]

    left_lines = [] # Positive slope (m < 0)
    right_lines = [] # Negative slope (m >= 0)

    for line in lines:
        for x1,y1,x2,y2 in line:

            if x1 == x2:
                pass # Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                print(f"m: {m}, c: {c}")

                if m < 0:
                    left_lines.append((m,c))
                elif m >= 0:
                    right_lines.append((m,c))

    left_line = [0]
    right_line = [0]

    if len(left_lines) > 0:
        left_line = np.mean(left_lines, axis=0)
    
    if len(right_lines) > 0:
        right_line = np.mean(right_lines, axis=0)

    lines = [left_line, right_line]
    significant_lines = [line for line in lines if len(line) != 1]

    for slope, intercept in significant_lines:

        # Getting complete height of image in y1
        rows, _ = image.shape[:2]
        y1 = int(rows)

        # Taking y2 upto 50% of actual height or 60% of y1
        y2= int(rows * 0.5)

        # x = (y-c)/m
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(image_copy, np.array([[[x1,y1,x2,y2]]]))
    
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(image_copy, pts = np.array([poly_vertices],'int32'), color = (0,255,0))
    return cv2.addWeighted(image, 0.6, image_copy, 0.4, 0)

def hough_lines(image, rho, theta, thresh, min_line_len, max_line_gap):
    """        
    Returns an image with hough lines drawn.
    """
    print(f"Drawing hough lines...")
    lines = cv2.HoughLinesP(image, rho, theta, thresh, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    line_img = calculate_poly(line_img,lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.1, beta=1., gamma=0):
    """    
    The result image is computed as follows:
    
    initial_image * alpha + image * beta + gamma (initial_image and image should be the same shape)
    """
    print(f"Applying weighted image...")
    lines_edges = cv2.addWeighted(initial_img, alpha, img, beta, gamma)
    return lines_edges

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.15, rows]
    top_left     = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6] 
    
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver