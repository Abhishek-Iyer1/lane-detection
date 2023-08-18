import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import typing


train: list = pickle.load(open("data/full_CNN_train.p", 'rb'))
test: list = pickle.load(open("data/full_CNN_labels.p", 'rb'))

image = train[200]

x_size: int = image.shape[0]
y_size: int = image.shape[1]
channels: int = image.shape[2]
color_thresh_img = np.array(np.copy(image))
image_bgr = np.array(np.copy(image))

gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray_thresh_img = np.array(np.copy(gray_img))

# Converting RGB Format to BGR
temp_channel = image_bgr[:,:,2]
image_bgr[:,:,2] = image_bgr[:,:,0]
image_bgr[:,:,0] = temp_channel

# Create area of interest
x_top = 40
x_bottom = 80
y_left = 0
y_right = 400

# Crop original image based on region of interest
roi_img = image[x_top:x_bottom, y_left:y_right]
roi_gray = gray_img[x_top:x_bottom, y_left:y_right]
roi_area = (x_bottom - x_top) * (y_right - y_left)
roi_mask = np.zeros([x_size, y_size, channels], dtype=np.uint8)
roi_mask[x_top:x_bottom, y_left:y_right, :] = [1, 1, 1]
roi_mask_gray = np.zeros([x_size, y_size], dtype=np.uint8)
roi_mask_gray[x_top:x_bottom, y_left:y_right] = 1

# Calculate averages for each channel to set the threshold accordingly
r_pixel_avg = sum(list(map(sum, roi_img[:,:,0]))) / roi_area
g_pixel_avg = sum(list(map(sum, roi_img[:,:,1]))) / roi_area
b_pixel_avg = sum(list(map(sum, roi_img[:,:,2]))) / roi_area
bgr_averages = (b_pixel_avg, g_pixel_avg, r_pixel_avg)

gray_avg = sum(list(map(sum, roi_gray))) / roi_area

# Set threshold values for B,G,R channels
# print(f"Averages = R: {b_pixel_avg}, G: {g_pixel_avg}, B: {r_pixel_avg}")
bgr_threshold = [0,0,0]
for avg, i in list(zip(bgr_averages, range(3))):
    bgr_threshold[i] = ((255 - avg) // 3) + avg
    # rgb_threshold[i] = 200

gray_threshold = ((255 - gray_avg) // 3) + gray_avg

# Mask image where pixels are below this value
color_threshold_mask = (image_bgr[:,:,0] < bgr_threshold[0]) | (image_bgr[:,:,1] < bgr_threshold[1]) | (image_bgr[:,:,2] < bgr_threshold[2])
color_thresh_img[color_threshold_mask] = [0,0,0]
color_thresh_img *= roi_mask

gray_threshold_mask = (gray_img < gray_threshold)
gray_thresh_img[gray_threshold_mask] = 0
gray_thresh_img *= roi_mask_gray

# print(f"color thresh img: {color_thresh_img}, roi mask: {roi_mask}")

# print(f"Thresholds = R: {bgr_threshold[0]}, G: {bgr_threshold[1]}, B: {bgr_threshold[2]}")

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
low_threshold = 100
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Pasted Code
# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
vertices = np.array([[(0,x_size),(0, 40), (160, 40), (y_size,x_size)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 2     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 4 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
lines_edges = cv2.polylines(lines_edges,vertices, True, (0,0,255), 3)

plt.imshow(edges, cmap="Greys_r")
plt.show()
plt.imshow(image)
plt.title("Input Image")
plt.show()
plt.imshow(lines_edges)
plt.title("Colored Lane line [In RED] and Region of Interest [In Blue]")
plt.show()