import pickle
import matplotlib.pyplot as plt
import sys

train: list = pickle.load(open("data/full_CNN_train.p", 'rb'))
test: list = pickle.load(open("data/full_CNN_labels.p", 'rb'))

if (len(sys.argv) == 3) and (int(sys.argv[2]) < len(train)):

    index = int(sys.argv[2])
    number_of_images = int(sys.argv[1])
    rows = 2
    columns = number_of_images

    fig = plt.figure()
    for i in range(1, number_of_images + 1):

        # Image from Train set

        fig.add_subplot(rows, columns, i)
        plt.imshow(train[index + i])
        plt.axis("off")
        plt.title(f"Train: {index + i}")

        # Corresponding Image from test set

        fig.add_subplot(rows, columns, number_of_images + i)
        plt.imshow(test[index + i])
        plt.axis("off")
        plt.title(f"Test: {index + i}")

    plt.show()

else:
    raise ValueError(f"\nArgument 1 is the amount of images to display: recommend  <10\nArgument 2 must be less than the length of the dataset: {len(train)}")