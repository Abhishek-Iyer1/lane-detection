import sys
import pickle
import matplotlib.pyplot as plt
from classical_utils import *

def run_classical_pipeline():
    """
    Description: 
        Requires image 
    """
    
    # Load train and test data
    train = np.array(pickle.load(open("data/full_CNN_train.p", 'rb')))
    labels = np.array(pickle.load(open("data/full_CNN_labels.p", 'rb')))

    # Find image number as given through cli from train data
    image = train[int(sys.argv[1])]

    # Run lane detection pipeline on the image
    images = lane_detection_classical(image)

    # Add the ground truth image to the list of images
    images.append(("Ground Truth", labels[int(sys.argv[1])]))

    # Plot results
    fig = plt.figure()

    for i, image_data in list(zip(range(1,len(images) + 1), images)):
        title, image = image_data
        fig.add_subplot(1, len(images), i)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image)

    plt.show()

def lane_detection_classical(image):
    """
    Description: 
        The whole pipeline for detecting lanes using classical approaches with opencv.
    """
    gray_image = grayscale(image)
    smoothed_image = gaussian_blur(image = gray_image, kernel_size = 5)
    canny_image = canny(image = smoothed_image, low_thresh = 180, high_thresh = 240)
    masked_image = apply_mask(image = canny_image, vertices = get_vertices(image))
    houghed_line_image = hough_lines(image = masked_image, rho = 1, theta = np.pi/180, thresh = 2, min_line_len = 4, max_line_gap = 5)
    final_output = weighted_img(line_image = houghed_line_image, initial_image=image, vertices= get_vertices(image), alpha=0.8, beta=1, gamma=0)
    
    return [("Gray Image", gray_image), ("Smoothed Image", smoothed_image), ("Canny Image", canny_image), ("Masked Image", masked_image), ("Hough Transform Image", houghed_line_image), ("Final Ouput", final_output)]

if __name__ == "__main__":
    run_classical_pipeline()