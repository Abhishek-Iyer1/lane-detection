# lane-detection

In this repository, we are aiming to explore traditional and smart ways to segment and detect lanes to assist with automated driving.

## Traditional Approach Pipeline
* Convert the image to grayscale
* Using thresholding to see as a baseline to see what percentage of the lanes we can already detect
* Cropping the image and creating an area of interest mask to focus our attention, since mounted cameras can be assumed to have approximately the same position.
* Using the Canny edge detection technique and tuning its parameters to get the best output within our area of interest
* Using Hough transform to detect line segments.

## Testing performance
* Test on test images
* Test real-time performance by applying to a 30 FPS video

## Displaying Outputs
* Images of Input, Grayscale, after Canny, after masking, and after hough transform to be added for visualising the steps of the pipeline.

## Deep Learning Pipeline
* Create and train a simple CNN model for performing real-time segmentation of lanes
* Can explore lightweight segmentation models if performance is lacking
  * UNET (implemented)
  * FPN
  * Different backbones such as VGG 16, Resnet-50, EfficientNet, Vision transformers

## Difficulties
* The classical pipeline struggles to adjust to different times of the day, curvature in lanes, and different colours of the lanes themselves.
* Facing issues to integrate checks for local GPU and use it for training models.
* Only able to load 6k images from the dataset.
