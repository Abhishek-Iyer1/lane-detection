import pickle
import matplotlib.pyplot as plt
import numpy as np
import typing
import cv2

train: list = pickle.load(open("data/full_CNN_train.p", 'rb'))
test: list = pickle.load(open("data/full_CNN_labels.p", 'rb'))

image = train[3000]

x_size: int = image.shape[0]
y_size: int = image.shape[1]
channels: int = image.shape[2]
color_thresh_img = np.array(np.copy(image))

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_thresh_img = np.array(np.copy(gray_img))

# Converting RGB Format to BGR
temp_channel = image[:,:,2]
image[:,:,2] = image[:,:,0]
image[:,:,0] = temp_channel

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
print(f"Averages = R: {b_pixel_avg}, G: {g_pixel_avg}, B: {r_pixel_avg}")
bgr_threshold = [0,0,0]
for avg, i in list(zip(bgr_averages, range(3))):
    bgr_threshold[i] = ((255 - avg) // 3) + avg
    # rgb_threshold[i] = 200

gray_threshold = ((255 - gray_avg) // 3) + gray_avg

# Mask image where pixels are below this value
color_threshold_mask = (image[:,:,0] < bgr_threshold[0]) | (image[:,:,1] < bgr_threshold[1]) | (image[:,:,2] < bgr_threshold[2])
color_thresh_img[color_threshold_mask] = [0,0,0]
color_thresh_img *= roi_mask

gray_threshold_mask = (gray_img < gray_threshold)
gray_thresh_img[gray_threshold_mask] = 0
gray_thresh_img *= roi_mask_gray

# print(f"color thresh img: {color_thresh_img}, roi mask: {roi_mask}")

print(f"Thresholds = R: {bgr_threshold[0]}, G: {bgr_threshold[1]}, B: {bgr_threshold[2]}")
plt.imshow(gray_thresh_img)
plt.show()