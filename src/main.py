import sys
import pickle
import tensorflow as tf
import numpy as np

from src.classical_utils import *
from unet import UNET
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report, confusion_matrix

def main():

    # Check for existing GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=15372)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Load train and test data
    train = np.array(pickle.load(open("data/full_CNN_train.p", 'rb')))
    labels = np.array(pickle.load(open("data/full_CNN_labels.p", 'rb')))

    # Train on subset of the dataset due to memory limitations
    train = train[0:6000]
    labels = labels[0:6000]

    # Reshape and Normalize training data
    train = np.reshape(train, (-1, 80, 160, 3)) / 255
    labels = np.reshape(labels, (-1, 80, 160, 1)) / 255

    # Generates train and test datasets from the original dataset after shuffling, splitting, and creating batches.
    train_dataset, test_dataset = prepare_dataset(train, labels, batch_size=32)

    # Training
    # my_unet = UNET(input_shape=image.shape, trainable=True, start_filters=16, name="trial unet")
    # my_unet.model.summary()
    # history = train_model(my_unet, train_dataset=train_dataset, test_dataset=test_dataset)
    # my_unet.model.save_weights("unet_weights.h5")

def prepare_dataset(train: list, labels: list, train_split: float = 0.8, batch_size=32):

    """
    Description:
        Generates a separate train dataset and test dataset depending on the train split. Shuffle, split, and create batches.

    Arguments:
    1. train = list with all the training images.
    2. labels = list of all the masks or labels for respective training images.
    3. train_split = percentage of dataset that should be included in the train dataset. The remaining will make up the test dataset.
    4. batch_size = the number of training images to be used together in one iteration.

    Returns:
    1. train_dataset = tf.data.Dataset object with the train images and labels.
    2. test_dataset = tf.data.Dataset object with the test images and labels.
    """

    num_elements = len(train)
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train), tf.convert_to_tensor(labels)))
    dataset = dataset.shuffle(num_elements)
    train_size = int(train_split * num_elements)
    test_size = num_elements - train_size
    train_dataset = dataset.skip(test_size)
    test_dataset = dataset.take(test_size)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)
    return train_dataset, test_dataset

def train_model(model: UNET, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset):

    """
    Description:
        Trains the UNET model and saves a weights file and a keras model folder which can be used for predicting.
        
    Arguments:
    1. model = instance of class UNET
    2. x_data = np array of all image training data
    3. y_data = np array of all segmented masks or label corresponding to the image training data

    Returns:
        None.
        
    NOTE: keras model folder saved in local directory and a weights file saved as well to save progress.
    """

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.00001, verbose=1)
    
    epochs = 200
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    loss = "binary_crossentropy"
    optimizer="adam"
    metrics=["accuracy"]
    model.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    if model.trainable:
        history = model.model.fit(
                    train_dataset,
                    validation_data=test_dataset,
                    epochs=epochs,
                    callbacks=callbacks
                )
        return history
    else:
        raise ValueError (f"smodel.trainable value is {model.trainable}. Please set the value to True in order to train the model.")

if __name__ == '__main__':
    main()

