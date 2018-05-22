#!/usr/bin/env python3

# Class code for YOLO

# Saffat Shams Akanda, <Your Name Here>, <Your Name Here>

import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Input
import cv2
import numpy as np
import os

EPOCHS = 1000


class YOLO:

    def __init__(self, input_size):
        self.input_size = input_size
        pass

    def customYOLOLoss(self):
        pass

    def predict(self):
        pass

    def train(self, x_train, y_train, x_val, y_val):
        tbCallBack = keras.callbacks.TensorBoard(log_dir="./Logs", write_graph=True)
        checkpointCallBack = keras.callbacks.ModelCheckpoint("./Checkpoints/model_and_weights.{epoch:02d}.hdf5",
                                                             period=5)

        self.model.fit(x_train, y_train, batch_size=128, epochs=EPOCHS, validation_data=(x_val, y_val),
                       callbacks=[tbCallBack, checkpointCallBack])

    def build_yolo_model(self):
        pass

    def build_tiny_yolo_model(self):
        self.inputLayer = Input(shape=(self.input_size, self.input_size, 3))

        # Layer 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(self.inputLayer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Layer 2
        x = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Layer 3
        x = Conv2D(16, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 4
        x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 5
        x = Conv2D(16, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 6
        x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Layer 7
        x = Conv2D(32, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 9
        x = Conv2D(32, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 10
        x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        # Layer 11
        x = Conv2D(64, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 12
        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 13
        x = Conv2D(64, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 14
        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 15
        x = Conv2D(128, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 16
        x = Conv2D(1000, (1, 1), strides=(1, 1), padding="same", name="conv1", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = keras.layers.AveragePooling2D()(x)
        x = keras.layers.Softmax()

        self.model = keras.Model(self.inputLayer, x)
        self.model.compile(optimizer="adam", loss=self.customYOLOLoss(), metrics=["accuracy"])


