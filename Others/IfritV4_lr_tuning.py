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
batch_size = 128                    # indicate the actual value of batch_size used in training
early_patience = 15                 # patience for early stopping in the training
result_dict = {}                    # dictionary that contains results for each k-cross validation done
hypermodel = None                   # contain the CNN model, default value is None

# ---- dataset variables ----
classes = []                        # the label associated with each class will be the position that the class name will have in this array
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
    
# method for create and plot the confusion metrix of the model trained
def confusion_matrix():
    global test_image, test_label, hypermodel, classes # global variables references
    # create the confusion matrix, rows indicate the real class and columns indicate the predicted class 
    conf_matrix = np.zeros((len(classes),len(classes)))     # at begin values are 0
    
    test_loss, test_acc = hypermodel.evaluate(test_image, test_label)    
    
    predictions = hypermodel.predict(test_image)               # get the output for each sample of the test set
    # slide the prediction result and go to create the confusion matrix
    for i in range(len(test_image)):
        # test_label[i] indicate the real value of the label associated at the test_image[i] -> is real class (row)
        # predictions[i] indicate the class value predicted by the model for the test_image[i] -> is predicted class (column)
        # the values are in categorical format, translate in int
        conf_matrix[np.argmax(test_label[i])][np.argmax(predictions[i])] += 1                              # update value
        
    # do percentages of confusion matrix
    conf_matrix_perc = [[None for c in range(conf_matrix.shape[1])] for r in range(conf_matrix.shape[0])]  # define matrix
    
    for i in range(conf_matrix.shape[0]):                   # rows
        for j in range(conf_matrix.shape[1]):               # columns
            conf_matrix_perc[i][j] = " (" + str( round( (conf_matrix[i][j]/len(test_image))*100 ,2) ) + "%)"    # calculate percentage value
    
    # plot the confusion matrix
    rows = classes                                          # contain the label of the classes showed in the row values of rows          
    columns = classes                                       # contain the label of the classes showed in the row values of columns   

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    
    for i in range(len(rows)):                              # rows
        for j in range(len(columns)):                       # columns
            # give the value in the confusion matrix
            ax.text(x=j, y=i, s=str(str(conf_matrix[i][j])+conf_matrix_perc[i][j]),
                           ha="center", va="center", size='x-large')
            
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Real', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()   
# ------------------------------------ end: method for plot results ------------------------------------
#GPU_check()
# assign value to classes
for folder in list_dir_ds:                      # for each folder in DS
    classes.append(str(folder))                 # update classes
            
# load the whole dataset
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

# control check print of the sets
print("Dimension check print of the sets (measured in batch)")
print("val_train_dimension: ", batch_train_val_size)
print("Training set dimension: ",batch_train_size, " val set dimension: ", batch_val_size, " test set dimension: ", batch_test_size)        # dimensione in batch

# convert in np.array
numpy = test_data.as_numpy_iterator()               # return a numpy iterator

# slide the iterator, this is divided by batch
for batch in numpy:
    # each batch is divided in 2 list: one for images and one for labels
    #print("imm:" , batch[0], "label ", batch[1])                                           # check print of the whole batch divided by images and labels
    #print("dimension of images: ",len(batch[0]),"\ndimension of labels: ",len(batch[1]))   # check print for the length of the images and label in the batch, the number of images and labels must be equal to each other and must be batch size
    
    # slide the images of the current batch (first list batch[0])
    for img in batch[0]:
        test_image.append(img)              # add image to test_image 
    # slide the labels of the current batch (second list batch[1])
    for label in batch[1]:
        test_label.append(label)            # add correlated label to test_label

# convert in np.array
test_image = np.array(test_image)
test_label = np.array(test_label)
# control data print
print("test_image",len(test_image), test_image.shape)
print("test_label",len(test_label), test_label.shape)
print("Requied memory for images in test set: ",test_image.size * test_image.itemsize / 10**9," GB")
# preprocessing
#test_image = test_image.astype('float32') / 255                                         # normalization

"""
# check print of the first batch_size from training and validation set, tha batch size must be divisible by 4
# print first batch of training set
plt.figure(figsize=(9, 6))
for images, labels in train_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(16, batch_size // 16, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_val_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of training set')
plt.show()

# print first batch of validation set
plt.figure(figsize=(9, 6))
for images, labels in val_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(16, batch_size // 16, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_val_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of validation set')
plt.show()

# print first batch of test set
plt.figure(figsize=(9, 6))
for images, labels in test_data.take(1):
  for i in range(batch_size):
    ax = plt.subplot(16, batch_size // 16, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(test_data.class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.suptitle('First batch of test set')
plt.show()
"""
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     directory=log_dir,
                     project_name='INet4_LR')

checkpoint = ModelCheckpoint(filepath = path_check_point_model+"/check_I4Learn.hdf5", verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max

eStop = EarlyStopping(patience = early_patience, verbose = 1, restore_best_weights = True, monitor='val_loss')

#tuner.search(train_image, train_label, epochs=epochs, validation_split=val_set_split, callbacks=[eStop])
tuner.search(train_data, epochs=epochs, validation_data=val_data, callbacks=[checkpoint, eStop])

# Print the tuner value name
#print("Tuner value name: ",tuner.get_best_hyperparameters()[0].values)
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The hyperparameter search is complete. \
        The optimal number of dropout in the last layer is {best_hps.get('dropout')} and \
        the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")
 
# Build the model with the optimal hyperparameters and train it on the data 
hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[checkpoint, eStop])

plot_fit_result(history.history,0)                  # visualize the value for the fit - history.history is a dictionary - call method for plot train result

test_loss, test_acc = hypermodel.evaluate(test_data)
dict_metrics = {'loss': test_loss, 'accuracy': test_acc}                            # create a dictionary contain the metrics
plot_fit_result(dict_metrics,1)      
print("[test loss, test accuracy]:", test_loss, test_acc)

confusion_matrix()                                  # call method to obtain the confusion matrix


