import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def main():
    # Check if passed argument is a valid path
    if os.path.exists(sys.argv[1]):
        file_path = sys.argv[1]
    else:
        raise ValueError (f"{sys.argv[1]} is not a valid path, please check again.")
    
    # If it is a valid path, try to load it as a clip file.
    clip = VideoFileClip(file_path)
    # Output file path
    path_output = f"output_video.mp4"
    # Create the clip
    vid_clip: VideoFileClip = clip.fl_image(detect_lanes)
    vid_clip.write_videofile(path_output, audio=False)

def detect_lanes(image):

    # Get image ready for feeding into model
    small_img = resize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Load saved model
    model: tf.keras.Model = tf.keras.models.load_model("keras.model")

    # Make prediction on input image
    prediction = model.predict(small_img)

    # Reshape and upscale image to match original image
    prediction = np.reshape(prediction, (80, 160))
    lane_full_size = resize(prediction, (720, 1280))

    # Create a 3 channel lane image to blend the original image with
    lane_image = np.zeros_like(image)
    lane_image[:,:,1] = lane_full_size * 255

    # Blend the original image with the detected lanes
    image_weighted = cv2.addWeighted(image, 1, lane_image, 1, 0.0)

    return image_weighted


if __name__ == '__main__':
    main()