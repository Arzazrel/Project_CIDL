# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that reproduces the AlexNet model

description: network description at the end of the file
"""
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers

# class that implement the AlexNet model
class AlexNet:
    # constructor
    def __init__(self,class_number):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = 224                   # height of the images in input to CNN
        self.img_width = 224                    # width of the images in input to CNN
        self.img_channel = 3                    # channel of the images in input to CNN (RGB)
        
    # method for make the model of the CNN
    def make_model(self):
        self.model = models.Sequential()
        # 1st Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 2nd Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 3rd Conv layer (hasn't Max pooling)
        self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 4th Conv layer (hasn't Max pooling)
        self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 5th Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 1th dense layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        self.model.add(layers.Dropout(0.5))                                                         # dropout
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # 2nd dense layer
        self.model.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        self.model.add(layers.Dropout(0.5))                                                         # dropout
        self.model.add(layers.local_response_normalization(bias=2,depth_radius=5, alpha=10**-4, beta=0.75))     # LRN Normalisation
        # Output layer
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))                        
        
        self.model.summary()                                                                        # summary of the net
        
    # method for compile the model
    def compile_model(self):
        # compile rmsprop
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
"""
brief description:
    AlexNet is a CNN model designed by Alex Krizhevsky and Ilya Sutskever, under the supervision of Geoffrey Hinton. 
    AlexNet represented a fundamental breakthrough in image classification problems and won the ImageNet Large Scale Visual Recognition Challenge in 2012. 
    AlexNet has eight levels: the first five levels are convolutional, of which the first two use max-pooling, while the last three levels do not, being fully connected. The last three layers are simple fully connected layers.
    The network uses local Response Normalization (LRN) and the ReLU activation function, except for the last layer which uses Softmax activation function.
    With the exception of the last layer, the rest of the network was split into two copies, running separately on two GPUs. 
"""