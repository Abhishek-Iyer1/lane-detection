import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

from keras import Model

def predict_lanes():
    # Load train and test data
    train = np.array(pickle.load(open("data/full_CNN_train.p", 'rb')))
    labels = np.array(pickle.load(open("data/full_CNN_labels.p", 'rb')))

    # Find image number as given through cli from train data
    image = train[int(sys.argv[1])]
    ground_truth = labels[int(sys.argv[1])]

    # Load saved model
    my_unet: Model = tf.keras.models.load_model('keras.model')

    # Generate predictions
    image_reshaped = np.reshape(image, [1, 80, 160, 3])
    output_test = my_unet.predict(image_reshaped)
    output = np.reshape(output_test, (80, 160, 1))

    images = [
        (f"Train Example, image no. {sys.argv[1]}", image),
        (f"Ground Truth, image no. {sys.argv[1]}", ground_truth),
        (f"Predicted Output", output)
    ]

    # Plot results
    fig = plt.figure()
        
    for i, image_data in list(zip(range(1,len(images) + 1), images)):
        title, image = image_data
        fig.add_subplot(1, len(images), i)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image)

    plt.show()

if __name__ == '__main__':
    predict_lanes()