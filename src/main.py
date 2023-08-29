import sys
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from lane_detection_utils import *
from unet import UNET
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report, confusion_matrix

def main():

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

    # Find image number as given through cli from train data
    image = train[int(sys.argv[1])]

    train = train[0:6000]
    labels = labels[0:6000]
    # Reshape and Normalize training data
    train = np.reshape(train, (-1, 80, 160, 3)) / 255
    labels = np.reshape(labels, (-1, 80, 160, 1)) / 255

    train_dataset, test_dataset = prepare_dataset(train, labels, batch_size=32)

    print(f"Normalized: {train.shape, labels.shape, train[60][50][50][0]}")

    # # Run lane detection pipeline on the image
    # images = lane_detection_pipeline_opencv(image)

    # # Add the ground truth image to the list of images
    # images.append(("Test Label", labels[int(sys.argv[1])]))

    # # Plot results
    # fig = plt.figure()

    # for i, image_data in list(zip(range(1,len(images)), images)):
    #     title, image = image_data
    #     fig.add_subplot(1, len(images), i)
    #     plt.title(title)
    #     plt.axis("off")
    #     plt.imshow(image)

    # plt.show()

    # Training
    # my_unet = UNET(input_shape=image.shape, trainable=True, start_filters=16, name="trial unet")
    # my_unet.model.summary()
    # history = train_model(my_unet, train_dataset=train_dataset, test_dataset=test_dataset)
    # my_unet.model.save_weights("unet_weights.h5")

    # Prediction
    my_unet: Model = tf.keras.models.load_model('keras.model')
    image = np.reshape(image, [1, 80, 160, 3])
    output_test = my_unet.predict(image)
    output = np.reshape(output_test, (80, 160, 1))
    plt.imshow(output)
    plt.show()

    # image = train[5]
    # image = np.reshape(image, [1, 80, 160, 3])
    # print(f"Image Shape: {image.shape}")
    # output = my_unet.model.predict(image)
    # print(f"Output: {output}")
    # output = np.reshape(output, (80, 160, 1))
    # plt.imshow(output)
    # plt.show()

def prepare_dataset(train: list, labels: list, train_split: float = 0.8, batch_size=32):
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

    # Check for GPU
    print(tf.config.list_physical_devices('GPU'))
    print(get_available_devices())

    # tf.data.Dataset approach

    # Indices approach
    # shuffle_indices = np.random.permutation(x_data.shape[0])
    # train_indices, valid_indices= train_test_split(shuffle_indices, test_size=0.2, random_state=42, shuffle=False)
    # print(f"Length of Shuffle Indices: {len(shuffle_indices)}\nLength of Train Indices: {len(train_indices)}\nLength of Labels Indices: {len(valid_indices)}")

    # Classic approach
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, test_size=0.2)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.00001, verbose=1)
    
    epochs = 200
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    loss = "binary_crossentropy"
    optimizer="adam"
    metrics=["accuracy"]
    # print(f"{len(x_data[train_indices]), len(x_data[valid_indices]), len(y_data[train_indices]), len(y_data[valid_indices])}")
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

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

# Lane finding Pipeline
def lane_detection_pipeline_opencv(image):
    
    gray_image = grayscale(image)
    smoothed_image = gaussian_blur(image = gray_image, kernel_size = 5)
    canny_image = canny(image = smoothed_image, low_thresh = 180, high_thresh = 240)
    masked_image = apply_mask(image = canny_image, vertices = get_vertices(image))
    houghed_line_image = hough_lines(image = masked_image, rho = 1, theta = np.pi/180, thresh = 2, min_line_len = 4, max_line_gap = 5)
    final_output = weighted_img(line_image = houghed_line_image, initial_image=image, vertices= get_vertices(image), alpha=0.8, beta=1, gamma=0)
    
    return [("Gray Image", gray_image), ("Smoothed Image", smoothed_image), ("Canny Image", canny_image), ("Masked Image", masked_image), ("Hough Transform Image", houghed_line_image), ("Final Ouput", final_output)]

if __name__ == '__main__':
    main()

