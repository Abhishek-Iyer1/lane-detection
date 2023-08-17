# lane-detection

In this repository, we are aiming to explore traditional and smart ways to segment and detect lanes to assist with automated driving.

## Traditional Approach Pipeline
* Convert the image to grayscale
* Using thresholding to see as a baseline to see what percentage of the lanes can we already detect
* Cropping the image and creating an area of interest mask to focus our attention, since mounted cameras can be assumed to have approximately the same position.
* Using the Canny edge detection technique and tuning its parameters to get the best output within our area of interest
* Using Hough transform to detect line segments. 

## Deep Learning Pipeline
* Create and train a simple CNN model for performing real-time segmentation of lanes
* Can explore lightweight segmentation models if performance is lacking
  * UNET
  * FPN
