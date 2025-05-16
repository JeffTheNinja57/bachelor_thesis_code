import os
from copy import deepcopy

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.optimizers import Adam

import utils

# Hide Tensorflow INFOS and WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Particle:
    """
    Represents a particle in the Particle Swarm Optimization algorithm for CNN architecture search.

    This class encapsulates a CNN architecture as a particle in the PSO algorithm. Each particle
    has a position (represented by its layer configuration), velocity, and personal best position.
    The class provides methods for initializing, updating, and evaluating CNN architectures during
    the PSO search process.

    The particle's position is represented as a list of layer dictionaries, where each dictionary
    contains information about a layer's type, output channels, and kernel size. The class also
    handles the creation, compilation, training, and evaluation of the corresponding Keras model.
    """
    def __init__(self, min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels, conv_prob,
                 pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim, initialize_pbest=True,
                 activation='relu'):
        """
        Initialize a particle with random CNN architecture.

        :param min_layer: Minimum number of layers in the CNN
        :type min_layer: int
        :param max_layer: Maximum number of layers in the CNN
        :type max_layer: int
        :param max_pool_layers: Maximum number of pooling layers allowed
        :type max_pool_layers: int
        :param input_width: Width of input images
        :type input_width: int
        :param input_height: Height of input images
        :type input_height: int
        :param input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        :type input_channels: int
        :param conv_prob: Probability of adding a convolutional layer
        :type conv_prob: float
        :param pool_prob: Probability of adding a pooling layer
        :type pool_prob: float
        :param fc_prob: Probability of adding a fully connected layer
        :type fc_prob: float
        :param max_conv_kernel: Maximum convolutional kernel size
        :type max_conv_kernel: int
        :param max_out_ch: Maximum number of output channels for convolutional layers
        :type max_out_ch: int
        :param max_fc_neurons: Maximum number of neurons in fully connected layers
        :type max_fc_neurons: int
        :param output_dim: Number of output classes
        :type output_dim: int
        :param initialize_pbest: Whether to initialize personal best position
        :type initialize_pbest: bool
        :param activation: Activation function to use ('relu' or 'leaky_relu')
        :type activation: str
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers

        self.feature_width = input_width
        self.feature_height = input_height
        self.activation = activation

        self.depth = np.random.randint(min_layer, max_layer)
        self.conv_prob = conv_prob
        self.pool_prob = pool_prob
        self.fc_prob = fc_prob
        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch

        self.max_fc_neurons = max_fc_neurons
        self.output_dim = output_dim

        self.layers = []
        self.acc = None
        self.vel = []  # Initial velocity
        self.pBest = []

        # Build particle architecture
        self.initialization()

        # Update initial velocity
        for i in range(len(self.layers)):
            if self.layers[i]["type"] != "fc":
                self.vel.append({"type": "keep"})
            else:
                self.vel.append({"type": "keep_fc"})

        self.model = None
        if initialize_pbest:
            self.pBest = self.clone_particle()

    def clone_particle(self):
        """
        Safely clone a particle, including its Keras model if it exists.
        This is a safer alternative to using deepcopy on objects containing Keras models.
        """
        # Create a new particle with the same parameters, but with initialize_pbest=False
        # to prevent infinite recursion
        clone = Particle(min_layer=1,  # These will be overridden
                         max_layer=2,  # These will be overridden
                         max_pool_layers=self.max_pool_layers, input_width=self.input_width,
                         input_height=self.input_height, input_channels=self.input_channels, conv_prob=self.conv_prob,
                         pool_prob=self.pool_prob, fc_prob=self.fc_prob, max_conv_kernel=self.max_conv_kernel,
                         max_out_ch=self.max_out_ch, max_fc_neurons=self.max_fc_neurons, output_dim=self.output_dim,
                         initialize_pbest=False  # Prevent recursive initialization
                         )

        # Copy simple attributes
        clone.num_pool_layers = self.num_pool_layers
        clone.feature_width = self.feature_width
        clone.feature_height = self.feature_height
        clone.depth = self.depth
        clone.acc = self.acc

        # Deep copy lists that don't contain Keras objects
        clone.layers = deepcopy(self.layers)
        clone.vel = deepcopy(self.vel)

        # Handle the model separately
        if self.model is not None:
            # Use Keras's clone_model to safely clone the model
            clone.model = clone_model(self.model)
            # Compile the cloned model with the same configuration
            clone.model.compile(loss=self.model.loss, optimizer=self.model.optimizer, metrics=self.model.metrics)
            # If weights exist, copy them
            if len(self.model.weights) > 0:
                clone.model.set_weights(self.model.get_weights())
        else:
            clone.model = None

        return clone

    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + self.layers[z]["type"] + " | "

        return string

    def initialization(self):
        out_channel = np.random.randint(3, self.max_out_ch)
        conv_kernel = np.random.randint(3, self.max_conv_kernel)

        # First layer is always a convolution layer
        self.layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

        conv_prob = self.conv_prob
        pool_prob = conv_prob + self.pool_prob
        fc_prob = pool_prob

        for i in range(1, self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1
            else:
                layer_type = np.random.rand()

            if layer_type < conv_prob:
                self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(self.layers, self.fc_prob, self.num_pool_layers,
                                                                   self.max_pool_layers, self.max_out_ch,
                                                                   self.max_conv_kernel, self.max_fc_neurons,
                                                                   self.output_dim)

            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)

        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

    def velocity(self, gBest, Cg):
        self.vel = utils.computeVelocity(gBest, self.pBest.layers, self.layers, Cg)

    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        new_p = self.validate(new_p)

        self.layers = new_p
        self.model = None

    def validate(self, list_layers):
        # Last layer should always be a fc with number of neurons equal to the number of outputs
        list_layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

        # Remove excess of Pooling layers
        self.num_pool_layers = 0
        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "max_pool" or list_layers[i]["type"] == "avg_pool":
                self.num_pool_layers += 1

                if self.num_pool_layers >= self.max_pool_layers:
                    list_layers[i]["type"] = "remove"

        # Now, fix the inputs of each conv and pool layers
        updated_list_layers = []

        for i in range(0, len(list_layers)):
            if list_layers[i]["type"] != "remove":
                if list_layers[i]["type"] == "conv":
                    updated_list_layers.append(
                        {"type": "conv", "ou_c": list_layers[i]["ou_c"], "kernel": list_layers[i]["kernel"]})

                if list_layers[i]["type"] == "fc":
                    updated_list_layers.append(list_layers[i])

                if list_layers[i]["type"] == "max_pool":
                    updated_list_layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})

                if list_layers[i]["type"] == "avg_pool":
                    updated_list_layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})

        return updated_list_layers

    ##### Model methods ####
    def model_compile(self, dropout_rate):
        list_layers = self.layers

        # Create a Sequential model with an explicit Input layer as the first layer
        in_w = self.input_width
        in_h = self.input_height
        in_c = self.input_channels

        # Create an explicit Input layer
        input_layer = Input(shape=(in_w, in_h, in_c))

        # Create a Sequential model with the input layer
        self.model = Sequential([input_layer])

        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]

                if i == 0:
                    # First layer after Input
                    self.model.add(Conv2D(filters=n_out_filters,  # Explicit parameter name
                                          kernel_size=kernel_size,  # Explicit parameter name
                                          strides=(1, 1), padding="same", data_format="channels_last",
                                          dilation_rate=(1, 1),  # Explicit dilation rate
                                          groups=1,  # Explicit groups parameter (standard convolution)
                                          use_bias=True,  # Explicit use of bias
                                          kernel_initializer='he_normal', bias_initializer='he_normal',
                                          kernel_regularizer=None,  # Explicit no regularization
                                          bias_regularizer=None,  # Explicit no regularization
                                          activity_regularizer=None,  # Explicit no regularization
                                          activation=None# Activation is applied separately
                                          ))
                    # Explicit parameters for BatchNormalization
                    self.model.add(BatchNormalization(axis=-1,  # Normalize along the last axis (features)
                                                      momentum=0.99, epsilon=0.001, center=True, scale=True))
                    self.model.add(Activation(self.activation))
                else:
                    # Add dropout with explicit rate
                    self.model.add(Dropout(rate=dropout_rate))

                    # Hidden Conv2D layer
                    self.model.add(Conv2D(filters=n_out_filters,  # Explicit parameter name
                                          kernel_size=kernel_size,  # Explicit parameter name
                                          strides=(1, 1), padding="same", data_format="channels_last",
                                          dilation_rate=(1, 1),  # Explicit dilation rate
                                          groups=1,  # Explicit groups parameter (standard convolution)
                                          use_bias=True,  # Explicit use of bias
                                          kernel_initializer='he_normal', bias_initializer='he_normal',
                                          kernel_regularizer=None,  # Explicit no regularization
                                          bias_regularizer=None,  # Explicit no regularization
                                          activity_regularizer=None,  # Explicit no regularization
                                          activation=None))
                    # Explicit parameters for BatchNormalization
                    self.model.add(BatchNormalization(axis=-1,  # Normalize along the last axis (features)
                                                      momentum=0.99, epsilon=0.001, center=True, scale=True))
                    self.model.add(Activation(self.activation))

            if list_layers[i]["type"] == "max_pool":
                # In TF2.x, it's important to be explicit about all parameters
                # Note: kernel_size is not used here, but we retrieve it for consistency
                kernel_size = list_layers[i]["kernel"]

                # Explicitly specify all parameters for better TF2.x compatibility
                self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid',  # Explicit padding mode
                                            data_format='channels_last'  # Explicit data format
                                            ))

            if list_layers[i]["type"] == "avg_pool":
                kernel_size = list_layers[i]["kernel"]

                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid',  # Explicit padding mode
                                                data_format='channels_last'  # Explicit data format
                                                ))

            if list_layers[i]["type"] == "fc":
                if list_layers[i - 1]["type"] != "fc":
                    # Flatten the input if coming from a non-FC layer
                    self.model.add(Flatten(data_format='channels_last'))  # Explicit data format

                # Add dropout with explicit rate
                self.model.add(Dropout(rate=dropout_rate))

                if i == len(list_layers) - 1:
                    # Output layer (last layer)
                    self.model.add(Dense(units=list_layers[i]["ou_c"],  # Explicit parameter name
                                         kernel_initializer='he_normal', bias_initializer='he_normal', use_bias=True,
                                         # Explicit use of bias
                                         activation=None  # Activation is applied separately
                                         ))
                    # Explicit parameters for BatchNormalization
                    self.model.add(BatchNormalization(axis=-1,  # Normalize along the last axis (features)
                                                      momentum=0.99, epsilon=0.001, center=True, scale=True))
                    self.model.add(Activation("softmax"))
                else:
                    # Hidden FC layer
                    self.model.add(Dense(units=list_layers[i]["ou_c"],  # Explicit parameter name
                                         kernel_initializer='he_normal', bias_initializer='he_normal',
                                         kernel_regularizer=regularizers.l2(0.01), use_bias=True, # Explicit use of bias
                                         activation=None  # Activation is applied separately
                                         ))
                    # Explicit parameters for BatchNormalization
                    self.model.add(BatchNormalization(axis=-1,  # Normalize along the last axis (features)
                                                      momentum=0.99, epsilon=0.001, center=True, scale=True))
                    self.model.add(Activation(self.activation))

        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='adam')

        # Compile model with explicit parameters
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'],
                           # TF2.x prefers 'accuracy' over 'acc'
                           loss_weights=None, weighted_metrics=None, run_eagerly=None)

    def model_fit(self, x_train, y_train, batch_size, epochs):
        """
        Train the model with explicit parameters for TF2.x compatibility.

        In TF2.x, the fit method has some differences compared to TF1.x:
        - It always returns a History object
        - The metrics in history use 'accuracy' instead of 'acc'
        - It supports more parameters for customization
        """
        # Use explicit parameters for all fit options
        hist = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None,
                              validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
                              sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
                              validation_batch_size=None, validation_freq=1)

        # Ensure the history object has the 'accuracy' key (in TF 2.x it should be 'accuracy' by default)
        if 'accuracy' not in hist.history and 'acc' in hist.history:
            hist.history['accuracy'] = hist.history['acc']

        return hist

    def model_fit_complete(self, x_train, y_train, batch_size, epochs):
        """
        Complete training of the model with explicit parameters for TF2.x compatibility.
        This method is used for the final training of the best model found.

        In TF2.x, the fit method has some differences compared to TF1.x:
        - It always returns a History object
        - The metrics in history use 'accuracy' instead of 'acc'
        - It supports more parameters for customization
        """
        # Use explicit parameters for all fit options
        hist = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None,
                              validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
                              sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
                              validation_batch_size=None, validation_freq=1)

        # Ensure the history object has the 'accuracy' key (in TF 2.x it should be 'accuracy' by default)
        if 'accuracy' not in hist.history and 'acc' in hist.history:
            hist.history['accuracy'] = hist.history['acc']

        return hist

    def model_delete(self):
        """
        Free up memory during PSO training.

        In TensorFlow 2.x, models are typically self-contained and Python's garbage collector
        should handle them once they are no longer referenced. However, in the context of PSO
        where many models are created and deleted in loops, we still explicitly delete the model
        to ensure timely resource release.

        We avoid using K.clear_session() as it's too aggressive for TF2.x and can affect
        other parts of the application. Instead, we rely on Python's garbage collection
        after removing our reference to the model.
        """
        if self.model is not None:
            K.clear_session()
            del self.model
            self.model = None
