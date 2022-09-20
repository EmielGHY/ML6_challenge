#!/usr/bin/env python3

"""Model to classify draft beers

This file contains all the model information: the training steps, the batch
size and the model itself.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value.
    """
    return 256

def get_epochs():
    """Returns number of epochs that will be used by your solution.
    It is recommended to change this value.
    """
    return 50

def solution(input_layer):
    """Returns a compiled model.

    This function is expected to return a model to identity the different beers.
    The model's outputs are expected to be probabilities for the classes and
    and it should be ready for training.
    The input layer specifies the shape of the images. The preprocessing
    applied to the images is specified in data.py.

    Add your solution below.

    Parameters:
        input_layer: A tf.keras.layers.InputLayer() specifying the shape of the input.
            RGB colored images, shape: (width, height, 3)
    Returns:
        model: A compiled model
    """

    num_classes = 5
    model = Sequential([
        input_layer,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['BinaryAccuracy'])

    # TODO: Code of your solution
    model = model

    # TODO: Return the compiled model
    return model
