# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: 
    program that does the test of googlenet
"""
# general
import random
import time
import math
import os
import tensorboard
# for model, tensorflow, keras and numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as back
import keras_tuner as kt
# for set sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
# for plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# for image visualization
import cv2
import PIL
from PIL import ImageTk
from PIL import Image
# for tensorflow-GPU
from tensorflow.python.client import device_lib 

# ------------------------------------ start: global var ------------------------------------
img_height = 224                    # height of the images in input to CNN
img_width = 224                     # width of the images in input to CNN
img_channel = 3                     # channel of the images in input to CNN     
epochs = 5                        # number of epochs of the training
batch_size = 128                    # indicate the actual value of batch_size used in training
early_patience = 15                 # patience for early stopping in the training

# ---- dataset variables ----
classes = []                        # the label associated with each class will be the position that the class name will have in this array
train_image = []                    # contain the images choosen as train set in fit
train_label = []                    # contain the labels of the images choosen as train set in fit
val_img = []                        # contain the images choosen as validation set in fit
val_label = []                      # contain the labels of the images choosen as validation set in fit
test_image = []                     # contain the images choosen as test set in evaluation
test_label = []                     # contain the labels of the images choosen as test set in evaluation
test_set_split = 0.2                # test set size as a percentage of the whole dataset
val_set_split = 0.2                 # validation set size as a percentage of the training set

# ---- path variables ----
path_dir_ds = os.path.join("Dataset","new_ds","Train_DS")                       # folder in which there are the image ds for training
path_ds = os.path.join(os.pardir,path_dir_ds)                                   # complete folder to reach the ds -- P.S. For more detail read note 0, please (at the end of the file) 
list_dir_ds = os.listdir(path_ds)                                               # list of the folders that are in the DS, one folder for each class
path_dir_model = "Model"                        # folder in which there are saved the CNN model
path_check_point_model = os.path.join(os.pardir,path_dir_model,"train_hdf5")  # folder in which there are saved the checkpoint for the model training
log_dir = os.path.join(path_check_point_model,"logs")
# ------------------------------------ end: global var ------------------------------------

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

inp = layers.Input(shape=(img_width, img_height, img_channel))       # input

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

# first auxiliary classifier
aux_1 = layers.AveragePooling2D((5, 5), strides=3)(seq_0)
aux_1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux_1)
aux_1 = layers.Flatten()(aux_1)
aux_1 = layers.Dense(1024, activation='relu')(aux_1)
aux_1 = layers.Dropout(0.7)(aux_1)
aux_1 = layers.Dense(2, activation='softmax',name = "aux_1")(aux_1)         # aux output layer

# seq_1: is the second part of the CNN network starting with the end of seq_0 and ending with the second auxiliary classifier
seq_1 = inception_mod(seq_0, fil_1x1=160, fil_1x1_3x3=112, fil_3x3=224, fil_1x1_5x5=24, fil_5x5=64, fil_m_pool=64)
seq_1 = inception_mod(seq_1, fil_1x1=128, fil_1x1_3x3=128, fil_3x3=256, fil_1x1_5x5=24, fil_5x5=64, fil_m_pool=64)
seq_1 = inception_mod(seq_1, fil_1x1=112, fil_1x1_3x3=144, fil_3x3=288, fil_1x1_5x5=32, fil_5x5=64, fil_m_pool=64)

# second auxiliary classifier
aux_2 = layers.AveragePooling2D((5, 5), strides=3)(seq_1)
aux_2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux_2)
aux_2 = layers.Flatten()(aux_2)
aux_2 = layers.Dense(1024, activation='relu')(aux_2)
aux_2 = layers.Dropout(0.7)(aux_2) 
aux_2 = layers.Dense(2, activation='softmax',name = "aux_2")(aux_2)         # aux output layer

# seq_2: is the last part of the CNN network starting with the end of seq_1 and ending with the end of CNN
seq_2 = inception_mod(seq_1, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
seq_2 = layers.MaxPooling2D(3, strides=2)(seq_2)
seq_2 = inception_mod(seq_2, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
seq_2 = inception_mod(seq_2, fil_1x1=384, fil_1x1_3x3=192, fil_3x3=384, fil_1x1_5x5=48, fil_5x5=128, fil_m_pool=128)
seq_2 = layers.GlobalAveragePooling2D()(seq_2)
seq_2 = layers.Dropout(0.4)(seq_2)
out = layers.Dense(2, activation='softmax',name = "out")(seq_2)           # output layer

model = Model(inputs = inp, outputs = [out, aux_1, aux_2])             # assign the CNN in model
model.summary()
print("output del modello: ",model.outputs)

model.compile(optimizer='adam', 
      loss=[losses.categorical_crossentropy,
            losses.categorical_crossentropy,
            losses.categorical_crossentropy],
      loss_weights=[1, 0.3, 0.3],
      metrics=['accuracy'])

# load the whole dataset and divide it in two split (change 'seed' to change the division and shuffle of the data)
# the two sets are already divided into images and labels, the labels are in categorical format
train_val_data , test_data = keras.utils.image_dataset_from_directory(
                  directory=path_ds,
                  labels= 'inferred',
                  label_mode='categorical',
                  color_mode='rgb',
                  batch_size=batch_size,
                  validation_split = test_set_split,
                  subset = "both",
                  seed=777,
                  shuffle=True,
                  image_size=(img_width, img_height) )

# size of the subdivisions of the various sets of the entire dataset, measured in batch size
batch_train_val_size = len(train_val_data)                              # batch size of training and validation set together
batch_train_size = int( (1 - val_set_split) * batch_train_val_size)     # batch size of training set
batch_val_size = batch_train_val_size - batch_train_size                # batch size of validation set
batch_test_size = len(test_data)                                        # batch size of test set

# division of data into sets
train_data = train_val_data.take(batch_train_size)
val_data = train_val_data.skip(batch_train_size)

# control check print of the sets
print("Dimension check print of the sets (measured in batch)")
print("val_train_dimension: ", batch_train_val_size)
print("Training set dimension: ",batch_train_size, " val set dimension: ", batch_val_size, " test set dimension: ", batch_test_size)        # dimensione in batch

# convert in np iterator
numpy_train = train_data.as_numpy_iterator()               # return a numpy iterator
numpy_val = val_data.as_numpy_iterator()               # return a numpy iterator
numpy_test = test_data.as_numpy_iterator()               # return a numpy iterator

# slide the iterator, this is divided by batch
for batch in numpy_train:
    # each batch is divided in 2 list: one for images and one for labels
    #print("imm:" , batch[0], "label ", batch[1])                                           # check print of the whole batch divided by images and labels
    #print("dimension of images: ",len(batch[0]),"\ndimension of labels: ",len(batch[1]))   # check print for the length of the images and label in the batch, the number of images and labels must be equal to each other and must be batch size
    
    # slide the images of the current batch (first list batch[0])
    for img in batch[0]:
        train_image.append(img)              # add image to test_image 
    # slide the labels of the current batch (second list batch[1])
    for label in batch[1]:
        train_label.append(label)            # add correlated label to test_label
# convert in np.array (necessary to have the confusion matrix in the format that I made and used)
train_image = np.array(train_image)
train_label = np.array(train_label)
train_image = train_image.astype('float32') / 255                                         # normalization
train_label = train_label.astype('float32')

# slide the iterator, this is divided by batch
for batch in numpy_val:
    # each batch is divided in 2 list: one for images and one for labels
    #print("imm:" , batch[0], "label ", batch[1])                                           # check print of the whole batch divided by images and labels
    #print("dimension of images: ",len(batch[0]),"\ndimension of labels: ",len(batch[1]))   # check print for the length of the images and label in the batch, the number of images and labels must be equal to each other and must be batch size
    
    # slide the images of the current batch (first list batch[0])
    for img in batch[0]:
        val_img.append(img)              # add image to test_image 
    # slide the labels of the current batch (second list batch[1])
    for label in batch[1]:
        val_label.append(label)            # add correlated label to test_label
# convert in np.array (necessary to have the confusion matrix in the format that I made and used)
val_img = np.array(val_img)
val_label = np.array(val_label)
val_img = val_img.astype('float32') / 255                                         # normalization
val_label = val_label.astype('float32')

# slide the iterator, this is divided by batch
for batch in numpy_test:
    # each batch is divided in 2 list: one for images and one for labels
    #print("imm:" , batch[0], "label ", batch[1])                                           # check print of the whole batch divided by images and labels
    #print("dimension of images: ",len(batch[0]),"\ndimension of labels: ",len(batch[1]))   # check print for the length of the images and label in the batch, the number of images and labels must be equal to each other and must be batch size
    
    # slide the images of the current batch (first list batch[0])
    for img in batch[0]:
        test_image.append(img)              # add image to test_image 
    # slide the labels of the current batch (second list batch[1])
    for label in batch[1]:
        test_label.append(label)            # add correlated label to test_label

# convert in np.array (necessary to have the confusion matrix in the format that I made and used)
test_image = np.array(test_image)
test_label = np.array(test_label)
test_image = test_image.astype('float32') / 255                                         # normalization
test_label = test_label.astype('float32')
# control data print
print("test_image",len(test_image), test_image.shape)
print("test_label",len(test_label), test_label.shape)
print("Requied memory for images in test set: ",test_image.size * test_image.itemsize / 10**9," GB")

checkpoint = ModelCheckpoint(filepath = path_check_point_model+'/prove_Google_.hdf5', verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max

eStop = EarlyStopping(patience = early_patience, verbose = 1, restore_best_weights = True, monitor='val_loss')

#history = model.fit(train_val_data, batch_size=batch_size, epochs=epochs)
#history = model.fit(train_val_data, validation_data = test_data ,batch_size=batch_size, epochs=epochs)
#history = model.fit(train_image,{'out':train_label, 'aux_1':train_label, 'aux_2':train_label},validation_data=(val_img,[val_label,val_label,val_label]),epochs=epochs,callbacks = [checkpoint, eStop])
history = model.fit(train_image,train_label,validation_data=(val_img,val_label),epochs=epochs,callbacks = [checkpoint, eStop])
print("Item di history", history.history)

# plot loss
fig = plt.figure()
fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # force the label of  number of epochs to be integer
plt.plot(history.history['loss'],'--', color="black", label = 'gen')
plt.plot(history.history['out_loss'],'--r', label = 'out')
plt.plot(history.history['aux_1_loss'],'--g', label = 'aux1')
plt.plot(history.history['aux_2_loss'],'--b', label = 'aux2')
plt.title(str("Loss"))              # plot title
plt.xlabel("# Epochs")              # x axis title
plt.ylabel("Value")                 # y axis title
plt.legend(loc='upper center')
plt.show()

gen_loss , out_loss, aux1_loss, aux2_loss, out_acc, aux1_acc, aux2_acc = model.evaluate(test_data)            # evaluate the model with test data
print("Result: ")
print("Total loss: ", gen_loss,"\nout loss: ", out_loss, "\naux1 loss: ", aux1_loss, "\naux2 loss: ", aux2_loss, "\nOut acc: ", out_acc, "\naux1 acc: ", aux1_acc, "\naux2 loss: ", aux2_acc)

predictions = model.predict(test_image)               # get the output for each sample of the test set
print("Predizione: ",len(predictions),len(predictions[0]))
"""
# plot loss
fig = plt.figure()
fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # force the label of  number of epochs to be integer
plt.plot(gen_loss,'--b', label = 'gen')
plt.plot(out_loss,'--r', label = 'out')
plt.plot(aux1_loss,'--g', label = 'aux1')
plt.plot(aux2_loss,'--b', label = 'aux2')
plt.title(str("Loss"))               # plot title
plt.xlabel("# Epochs")              # x axis title
plt.ylabel("Value")                 # y axis title
plt.legend(loc='upper center')
plt.show()
"""