# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that will contain the various versions of CNN made for fire_detection problem

description: network description at the end of the file in the note
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Model

# class that implement the IfriNet models
class IfriNet:
    # constructor
    def __init__(self,class_number):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = 224                   # height of the images in input to CNN
        self.img_width = 224                    # width of the images in input to CNN
        self.img_channel = 3                    # channel of the images in input to CNN (RGB)

    # method for make the models of the CNN. 'version_model' indicate the version of the CNN Ifrit
    def make_model(self,version_model):
        if version_model == 1:                  # first verstion, for more information 
            self.model = models.Sequential()                                   # rete del modello
            self.model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D((3, 3)))
            self.model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu'))
            self.model.add(layers.MaxPooling2D((3, 3)))
            self.model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu'))
            self.model.add(layers.MaxPooling2D((3, 3)))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.2))
            self.model.add(layers.Dense(self.class_number, activation='softmax'))