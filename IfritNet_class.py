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

# ------------------------------------ start: utility methods ------------------------------------

# utility function to implement the inception module in which 1×1, 3×3, 5×5 convolution and 3×3 max pooling are executed in parallel and their output is merged.
# in_net: is the input , fil_1x1: is the number of filters of conv 1x1 layer, the same for other similar fil
# fil_1x1_3x3: is the number of filters of the 1x1 reduction convolutionary layer before conv 3x3 and so on for others similar fil
# fil_m_pool: is the number of filter of the 1x1 convolutionary layer after max pooling 
def inception_mod(in_net, fil_1x1, fil_1x1_3x3, fil_3x3, fil_1x1_5x5, fil_5x5, fil_m_pool):
    # four parallel path
    
    path1 = layers.Conv2D(filters=fil_1x1, kernel_size=(1, 1), padding='same', activation='relu')(in_net)       # conv 1x1
    
    path2 = layers.Conv2D(filters=fil_1x1_3x3, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path2 = layers.Conv2D(filters=fil_3x3, kernel_size=(1, 1), padding='same', activation='relu')(path2)        # conv 3x3
    
    path3 = layers.Conv2D(filters=fil_1x1_5x5, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path3 = layers.Conv2D(filters=fil_5x5, kernel_size=(1, 1), padding='same', activation='relu')(path3)        # conv 5x5
    
    path4 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(in_net)                          # max pool
    path4 = layers.Conv2D(filters=fil_m_pool, kernel_size=(1, 1), padding='same', activation='relu')(path4)     # conv 1x1 to reduce
    
    return tf.concat([path1, path2, path3, path4], axis=3)                  # merge of the different path

# ------------------------------------ end: utility methods ------------------------------------

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
        
        if version_model == 1:                      # first verstion, for more information see Note 1 at the end of the file
            self.model = models.Sequential()                                   # rete del modello
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
        elif version_model == 2:                    # second verstion, for more information see Note 2 at the end of the file
            self.model = models.Sequential()
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 4th Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
        elif version_model == 3:                    # third verstion, for more information see Note 3 at the end of the file
            self.model = models.Sequential()
            # 1st Conv layer
            self.model.add(layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 3rd Conv layer
            self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 4th Conv layer
            self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            self.model.add(layers.Flatten())
            # 1th dense layer
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
            # 2nd dense layer
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dropout(0.3))                                                         # dropout
            # Output layer
            self.model.add(layers.Dense(self.num_classes, activation='softmax'))
            
        elif version_model == 4:                    # third verstion, for more information see Note 3 at the end of the file

            """
            # vers 1
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
            seq_0 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inp)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=96, fil_3x3=128, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=32)
            seq_0 = inception_mod(seq_0, fil_1x1=128, fil_1x1_3x3=128, fil_3x3=192, fil_1x1_5x5=32, fil_5x5=96, fil_m_pool=64)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=192, fil_1x1_3x3=96, fil_3x3=208, fil_1x1_5x5=16, fil_5x5=48, fil_m_pool=64)
            
            seq_2 = layers.GlobalAveragePooling2D()(seq_0)
            seq_2 = layers.Dropout(0.4)(seq_2)
            out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
            """
            """
            # vers 2
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
            seq_0 = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inp)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = layers.Conv2D(32, 1, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=96, fil_3x3=128, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=32)
            seq_0 = inception_mod(seq_0, fil_1x1=128, fil_1x1_3x3=128, fil_3x3=192, fil_1x1_5x5=32, fil_5x5=96, fil_m_pool=64)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=192, fil_1x1_3x3=96, fil_3x3=208, fil_1x1_5x5=16, fil_5x5=48, fil_m_pool=64)
            
            seq_2 = layers.GlobalAveragePooling2D()(seq_0)
            seq_2 = layers.Dropout(0.4)(seq_2)
            out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
            """
            """
            # vers 3
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
            seq_0 = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inp)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = layers.Conv2D(32, 1, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=32, fil_1x1_3x3=32, fil_3x3=64, fil_1x1_5x5=32, fil_5x5=32, fil_m_pool=32)
            seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=64, fil_3x3=128, fil_1x1_5x5=32, fil_5x5=64, fil_m_pool=64)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=128, fil_1x1_3x3=96, fil_3x3=128, fil_1x1_5x5=64, fil_5x5=64, fil_m_pool=64)
            
            seq_2 = layers.GlobalAveragePooling2D()(seq_0)
            seq_2 = layers.Dropout(0.4)(seq_2)
            out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
            """
            """
            # vers 4
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
            seq_0 = layers.Conv2D(16, 7, strides=2, padding='same', activation='relu')(inp)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = layers.Conv2D(32, 1, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=16, fil_1x1_3x3=16, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=32)
            seq_0 = inception_mod(seq_0, fil_1x1=32, fil_1x1_3x3=32, fil_3x3=64, fil_1x1_5x5=32, fil_5x5=64, fil_m_pool=64)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=64, fil_3x3=128, fil_1x1_5x5=64, fil_5x5=64, fil_m_pool=64)
            
            seq_2 = layers.GlobalAveragePooling2D()(seq_0)
            seq_2 = layers.Dropout(0.4)(seq_2)
            out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
            """
            
            # vers 5
            inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
            # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
            seq_0 = layers.Conv2D(16, 7, strides=2, padding='same', activation='relu')(inp)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = layers.Conv2D(32, 1, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(seq_0)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=32, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=8, fil_5x5=32, fil_m_pool=32)
            seq_0 = inception_mod(seq_0, fil_1x1=32, fil_1x1_3x3=16, fil_3x3=64, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=64)
            seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
            seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=32, fil_3x3=128, fil_1x1_5x5=32, fil_5x5=64, fil_m_pool=32)
            
            seq_2 = layers.GlobalAveragePooling2D()(seq_0)
            seq_2 = layers.Dropout(0.4)(seq_2)
            out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
            
            self.model = Model(inputs = inp, outputs = out)             # assign the CNN in model
        
        self.model.summary()
            
    # method for compile the model
    def compile_model(self):
        # compile rmsprop
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    # method for return the model
    def return_model(self):
        return self.model

og0 = IfriNet(2)

#og0.make_model(1)                       # make model (IfriNet 1 architecture)
#og0.make_model(2)                       # make model (IfriNet 1 architecture)
#og0.make_model(3)                       # make model (IfriNet 1 architecture) 
og0.make_model(4)                       # make model (IfriNet 1 architecture) 
"""
-------- Notes --------
-- Note 1 --
 

-- Note 2 --

-- Note 3 --

"""