import sys
import pickle
import matplotlib.pyplot as plt
from lane_detection_utils import *
from unet import UNET
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

def main():
    # Load train and test data
    train: list = pickle.load(open("data/full_CNN_train.p", 'rb'))
    labels: list = pickle.load(open("data/full_CNN_labels.p", 'rb'))

    # Find image number as given through cli from train data
    image = train[int(sys.argv[1])]

    # Run lane detection pipeline on the image
    images = lane_detection_pipeline_opencv(image)

    # Add the ground truth image to the list of images
    images.append(("Test Label", labels[int(sys.argv[1])]))

    # # Plot results
    # fig = plt.figure()

    # for i, image_data in list(zip(range(1,len(images)), images)):
    #     title, image = image_data
    #     fig.add_subplot(1, len(images), i)
    #     plt.title(title)
    #     plt.axis("off")
    #     plt.imshow(image)

    # plt.show()

    my_unet = UNET(input_shape=image.shape, trainable=True, start_filters=16, name="trial unet")
    my_unet.model.summary()
    train_model(my_unet, x_data=train, y_data=labels)
    # image = train[5]
    # image = np.reshape(image, [1, 80, 160, 3])
    # print(f"Image Shape: {image.shape}")
    # output = my_unet.model.predict(image)
    # print(f"Output: {output}")
    # output = np.reshape(output, (80, 160, 1))
    # plt.imshow(output)
    # plt.show()

def train_model(model: UNET, x_data: list, y_data: list):
    
    x_train, x_test, y_train, y_test = train_test_split(np.array(x_data).reshape(-1, 80, 160, 3), np.array(y_data).reshape(-1, 80, 160, 1), train_size=0.8, test_size=0.2)
    # x_train = 
    # y_train = 
    # validation_data = 

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    
    batch_size = 32
    epochs = 200
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    validation_data = [x_test, y_test]
    model.train_model(x_train=x_train, 
                      y_train=y_train, 
                      validation_data=validation_data, 
                      batch_size=batch_size, 
                      epochs=epochs, 
                      callbacks=callbacks, 
                      optimizer="adam", 
                      loss="binary_crossentropy", 
                      metrics=["accuracy"])

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

