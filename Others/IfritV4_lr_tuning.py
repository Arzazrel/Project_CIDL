# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: 
    program that does the hypertuning for the learning rate in the IfritNet v4 model that achieved the best performance.
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
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
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
epochs = 100                        # number of epochs of the training
k = 10                              # number of the fold of the k-cross validation
batch_size = 32                     # indicate the actual value of batch_size used in training
early_patience = 20                 # patience for early stopping in the training
result_dict = {}                    # dictionary that contains results for each k-cross validation done
network = None                      # contain the CNN model, default value is None
truncate_set = False                # variable which indicates whether the sets (train, test,val) must be truncate or not when divided to batch_size

# ---- dataset variables ----
classes = []                        # the label associated with each class will be the position that the class name will have in this array
total_image_ds = []                 # contain the total image dataset
total_labels_ds = []                # contain the labels of the total image dataset
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

# method to check GPU device avaible and setting
def GPU_check():
    #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    #print(os.getenv('TF_GPU_ALLOCATOR'))
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    print("Memoria usata: ",print("",tf.config.experimental.get_memory_info('GPU:0')['current'] / 10**9))

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

# function that return the model build
def model_builder(hp):
  inp = layers.Input(shape=(img_width, img_height, img_channel))                       # input
  
  resc = layers.Rescaling(1./255)(inp)                                                  # normalization layer
  
  net = layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2,2), padding='same', activation='relu')(resc)      # first conv layer
  net = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))(net)                                                 # max pool
  
  net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # first inception layer
  net = layers.MaxPooling2D(3, strides=1)(net)                                                                    # max pool 
  
  net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # second inception layer
  net = inception_mod(net, fil_1x1=32, fil_1x1_3x3=16, fil_3x3=64, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=64)     # third inception layer
  net = layers.MaxPooling2D(3, strides=2)(net)                                                                    # max pool
  
  net = inception_mod(net, fil_1x1=64, fil_1x1_3x3=16, fil_3x3=128, fil_1x1_5x5=8, fil_5x5=16, fil_m_pool=16)     # fourth inception layer
  net = layers.GlobalAveragePooling2D()(net)                                                                      # avg pool
  
  net = layers.Dense(64, activation='relu')(net)                              # fully connect 
  # Tune the dropout, chose a value in a range from 0 to 0.5 with step of 0.1
  net = layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(net)                                   # dropout
  out = layers.Dense(2, activation='softmax')(net)             # output layer
  
  model = Model(inputs = inp, outputs = out)             # assign the CNN in model

  # Tune the learning rate for the optimizer, choose an optimal value from 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005 or 0.00001
  hp_learning_rate = hp.Choice('learning_rate', values=[5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model

# ------------------------------------ start: method for plot results ------------------------------------
# method to plot accuracy and loss. arc is a dictionary with the results, 'mode' if is '0': there are fit results, if is '1': there are evaluation results
def plot_fit_result(arc,mode):
    result_dict = {}                    # dict that will contain the results to plot with the correct label/title
    # check what results there are
    if mode == 0:                       # method called with fit results
        result_dict["loss (training set)"] = arc["loss"]                    # take loss values (training set)
        result_dict["accuracy (taining set)"] = arc["accuracy"]             # take accuracy values (training set)
        if arc.get("val_loss") is not None:                         # check if there are result of validation set
            result_dict["loss (validation set)"] = arc["val_loss"]          # take loss values (validation set)
            result_dict["accuracy (validation set)"] = arc["val_accuracy"]  # take accuracy values (validation set)
    elif mode == 1:                     # method called with evaluate results
        result_dict["loss (test set)"] = arc["loss"]                        # take loss values (test set)
        result_dict["accuracy (test set)"] = arc["accuracy"]                # take accuracy values (test set)
    # plot the results
    for k,v in result_dict.items():
        #print("chiave: ", k," value: ",v)
        plot(k,v)

# method to display a plot. 'title' is the tile of the plot, 'value_list' is a list of value to draw in the plot
def plot(title,value_list):
    fig = plt.figure()
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # force the label of  number of epochs to be integer
    plt.plot(value_list,'o-b')
    plt.title(str(title))               # plot title
    plt.xlabel("# Epochs")              # x axis title
    plt.ylabel("Value")                 # y axis title
    plt.show()
# ------------------------------------ end: method for plot results ------------------------------------
#GPU_check()
"""
# -------- load the dataset --------
for folder in list_dir_ds:                      # for each folder in DS
    classes.append(str(folder))                 # update classes
    index = classes.index(str(folder))          # take index of classes, is teh label of this class
    p = os.path.join(path_ds,folder)            # path of each folder
    #creating a collection with the available images
    for filename in os.listdir(p):                      # for each images on the current folder
        img = cv2.imread(os.path.join(p,filename))      # take current iamge
        if img is not None:                             # check image taken
            #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                resize_img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)  # resize the image
                total_image_ds.append(resize_img)                                   # add image to total_image_ds
                total_labels_ds.append(index)                                       # add correlated label to total_lael_ds
            else:
                total_image_ds.append(img)                                          # add image to total_image_ds
                total_labels_ds.append(index)                                       # add correlated label to total_lael_ds
        else:
            print("Errore nel caricare immagine ",filename)

# convert in np.array
total_image_ds = np.array(total_image_ds)
total_labels_ds = np.array(total_labels_ds)
# control data print
print("Num of classes: ",len(classes))
print("total_image_ds",len(total_image_ds), total_image_ds.shape)
print("total_labels_ds",len(total_labels_ds), total_labels_ds.shape)
print("Requied memory for images ds: ",total_image_ds.size * total_image_ds.itemsize / 10**9," GB")
# preprocessing
#total_image_ds = total_image_ds.astype('float32') / 255                                         # normalization
#total_labels_ds = to_categorical(total_labels_ds,num_classes=len(classes))                # transform label in categorical format
"""
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
batch_train_val_size = len(train_val_data)                              #
batch_train_size = int( (1 - val_set_split) * batch_train_val_size)
batch_val_size = batch_train_val_size - batch_train_size
batch_test_size = len(test_data)

# division of data into sets
train_data = train_val_data.take(batch_train_size)
val_data = train_val_data.skip(batch_train_size)

# control check print
print("Dimension check print of the sets (measured in batch)")
print("val_train_dimension: ", batch_train_val_size)
print("Training set dimension: ",batch_train_size, " val set dimension: ", batch_val_size, " test set dimension: ", batch_test_size)        # dimensione in batch

# check print of the first batch_size from training and validation set, tha batch size must be divisible by 4
# print first batch of training set
plt.figure(figsize=(7, 7))
for images, labels in train_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(4, batch_size // 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_val_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of training set')
plt.show()

# print first batch of validation set
plt.figure(figsize=(10, 10))
for images, labels in val_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(4, batch_size // 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_val_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of validation set')
plt.show()

# print first batch of test set
plt.figure(figsize=(10, 10))
for images, labels in test_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(4, batch_size // 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(test_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of test set')
plt.show()

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     directory=log_dir,
                     project_name='INet4_LR')

checkpoint = ModelCheckpoint(filepath = path_check_point_model+"/check_I4Learn.hdf5", verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max

eStop = EarlyStopping(patience = early_patience, verbose = 1, restore_best_weights = True, monitor='val_loss')

#tensorboard = keras.callbacks.TensorBoard(log_dir)
#tuner.search(train_image, train_label, epochs=epochs, validation_split=val_set_split, callbacks=[eStop])
tuner.search(train_data, epochs=epochs, validation_data=val_data, callbacks=[eStop])

# Print the tuner value name
#print("Tuner value name: ",tuner.get_best_hyperparameters()[0].values)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The hyperparameter search is complete. \
        The optimal number of dropout in the last layer is {best_hps.get('dropout')} and \
        the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")
 
# Build the model with the optimal hyperparameters and train it on the data 
hypermodel = tuner.hypermodel.build(best_hps)
#hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
hypermodel.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[checkpoint, eStop])

eval_result = hypermodel.evaluate(test_data)
print("[test loss, test accuracy]:", eval_result)
