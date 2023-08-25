from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.layers import concatenate, Input
from keras import Model

class UNET():
    def __init__(self, input_shape: tuple[int], trainable: bool, start_filters=16, name = 'default_unet'):
        """
        Description:
            Initializing a UNET model and returns a model object that can be trained.
            
        Arguments:
        1. input_shape = The input shape that the Input() layer of the model will accept.
        2. trainable = Boolean value to signify if the model is trainable
        3. start_filters = Set the value of the starting number of filters for the model. They will increase to 8x the starting number at the deepest layers.

        Returns:
            model = Returns the built model with initialized weights which can be configured and trained.
        """
        self.name = name
        self.input_shape = input_shape
        self.start_filters = start_filters
        self.trainable = trainable
        self.input_layer = Input(self.input_shape)
        self.output_layer = self.build_unet(input_layer=self.input_layer, starting_filters=self.start_filters)
        self.model = Model(self.input_layer, self.output_layer)
        return self.model

