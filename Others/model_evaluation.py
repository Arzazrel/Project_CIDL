# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: 
    file to do evaluation of a model. given a model and a dataset perform fit and evaluation with cross validation for each pair value
    from the parameters (batch_size and early patience) and shows the results (accuracy, loss and confusion matrix)
"""
# general
import random
import time
import math
import os
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
batch_size_list = [32,64,128]       # batch size for the training
batch_size = 32                     # indicate the actual value of batch_size used in training
early_patience = [10,15,20]         # patience for early stopping in the training
result_dict = {}                    # dictionary that contains results for each k-cross validation done
network = None                      # contain the CNN model, default value is None
truncate_set = False                # variable which indicates whether the sets (train, test,val) must be truncate or not when divided to batch_size
model_name = "AlexNet"           # name of the model to test

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
val_set_split = 0.1                 # validation set size as a percentage of the training set

# ---- path variables ----
path_dir_ds = os.path.join("Dataset","new_ds","Train_DS")                       # folder in which there are the image ds for training
path_ds = os.path.join(os.pardir,path_dir_ds)                                   # complete folder to reach the ds -- P.S. For more detail read note 0, please (at the end of the file) 
list_dir_ds = os.listdir(path_ds)                                               # list of the folders that are in the DS, one folder for each class
path_dir_model = "Model"                        # folder in which there are saved the CNN model
path_check_point_model = os.path.join(os.pardir,path_dir_model,"train_hdf5")  # folder in which there are saved the checkpoint for the model training
# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: utility methods ------------------------------------
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

def make_model():
    # ---- AlexNet Model ----
    if model_name == "AlexNet":
        network = models.Sequential()
        # 1st Conv layer (has Max pooling)
        network.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd Conv layer (has Max pooling)
        network.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 3rd Conv layer (hasn't Max pooling)
        network.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 4th Conv layer (hasn't Max pooling)
        network.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 5th Conv layer (has Max pooling)
        network.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 1th dense layer
        network.add(layers.Flatten())
        network.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        network.add(layers.Dropout(0.5))                                                         # dropout
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 2nd dense layer
        network.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        network.add(layers.Dropout(0.5))                                                         # dropout
        network.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # Output layer
        network.add(layers.Dense(self.num_classes, activation='softmax'))       

        network.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    # ---- IfriNet Models ----
    elif model_name == "Ifrinet_v1":
        network = models.Sequential()                                   # rete del modello
        # 1st Conv layer
        network.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd Conv layer
        network.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 3rd Conv layer
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        network.add(layers.Flatten())
        # 1th dense layer
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd dense layer
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        # Output layer
        network.add(layers.Dense(self.num_classes, activation='softmax'))

        # compile Adam
        network.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == "Ifrinet_v2":
        network = models.Sequential()
        # 1st Conv layer
        network.add(layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd Conv layer
        network.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 3rd Conv layer
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 4th Conv layer
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        network.add(layers.Flatten())
        # 1th dense layer
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd dense layer
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        # Output layer
        network.add(layers.Dense(self.num_classes, activation='softmax'))

        # compile Adam
        network.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == "Ifrinet_v3":
        network = models.Sequential()
        # 1st Conv layer
        network.add(layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(3,3), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd Conv layer
        network.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 3rd Conv layer
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 4th Conv layer
        network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        network.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='valid'))       # Max pooling
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        network.add(layers.Flatten())
        # 1th dense layer
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        network.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd dense layer
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dropout(0.3))                                                         # dropout
        # Output layer
        network.add(layers.Dense(self.num_classes, activation='softmax'))

        # compile Adam
        network.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    elif model_name == "Ifrinet_v4":
        inp = layers.Input(shape=(img_width, img_height, img_channel))                       # input
    
        net = layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(2,2), padding='same', activation='relu')(inp)      # first conv layer
        net = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))(net)                                                 # max pool
    
        net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # first inception layer
        net = layers.MaxPooling2D(3, strides=1)(net)                                                                    # max pool 
    
        net = inception_mod(net, fil_1x1=16, fil_1x1_3x3=8, fil_3x3=32, fil_1x1_5x5=16, fil_5x5=64, fil_m_pool=32)      # second inception layer
        net = inception_mod(net, fil_1x1=32, fil_1x1_3x3=16, fil_3x3=64, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=64)     # third inception layer
        net = layers.MaxPooling2D(3, strides=2)(net)                                                                    # max pool
    
        net = inception_mod(net, fil_1x1=64, fil_1x1_3x3=16, fil_3x3=128, fil_1x1_5x5=8, fil_5x5=16, fil_m_pool=16)     # fourth inception layer
        net = layers.GlobalAveragePooling2D()(net)                                                                      # avg pool
    
        net = layers.Dense(64, activation='relu')(net)                              # fully connect 
        net = layers.Dropout(0.3)(net)                                              # dropout
        out = layers.Dense(len(classes), activation='softmax')(net)             # output layer
    
        network = Model(inputs = inp, outputs = out)             # assign the CNN in model
    
        # compile Adam
        network.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    return network
    
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
# ------------------------------------ end: utility methods ------------------------------------

# ------------------------------------ start: generetor function ------------------------------------
# explanation: for large dataset with large image or big batch size the memory memory may not be sufficient. 
#              To avoid memory overflow, the sets are supplied in batches via yeld istruction.
# define generator function to do the training set
def generator_train():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(train_image) % batch_size)     # check if the division by batch_size produce rest
        #print("Training test rest: ",rest)
    else:
        rest = batch_size                                       # set always truncated
    #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
    for idx in range(len(train_image)):                         # organize the sample in batch
        # take one image and the corresponding labels
        img = train_image[idx]                              
        label = train_label[idx]
        # add new element and convert to TF tensors
        img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
            # add this sample for the future (sample in the rest)
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(train_image) - 1):                       # check if the set is finished, last batch
            #print("arrivato alla fine")
            if rest != batch_size:                              # check if there are rest to fix
                #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                #print("Generator training set, arrivato a posizione: ",idx)
                yield img_tensor, label_tensor                  # return the last batch

# define generator function to do the validation set
def generator_val():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(val_img) % batch_size)         # check if the division by batch_size produce rest
        #print("Training test rest: ",rest)
    else:
        rest = batch_size                                       # set always truncated
    
    for idx in range(len(val_img)):                             # organize the sample in batch
        # take one image and the corresponding mask
        img = val_img[idx]
        label = val_label[idx]
        # add new element and convert to TF tensors
        img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(val_img) - 1):                           # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield img_tensor, label_tensor                  # return the last batch
        
# define generator function to do the test set
def generator_test():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(test_image) % batch_size)      # check if the division by batch_size produce rest
        #print("Training test rest: ",rest)
    else:
        rest = batch_size                                       # set always truncated
    
    for idx in range(len(test_image)):                          # organize the sample in batch
        # extract one image and the corresponding mask
        img = test_image[idx]
        label = test_label[idx]

        # add new element and convert to TF tensors
        img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(test_image) - 1):                        # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield img_tensor, label_tensor                  # return the last batch

# ------------------------------------ end: generetor function ------------------------------------

GPU_check()
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

# ----- preprocessing and reshape ----
# shuffle the data
data = list(zip(total_image_ds, total_labels_ds))   # organize img and label in zip 
np.random.shuffle(data)                             # shuffle zip
img, label = zip(*data)                             # return image and label shuffled
# convert in np.array
total_image_ds = np.array(img)
total_labels_ds = np.array(label)

lunghezza = len(total_image_ds)                                                     # take the len of the dataset
image_ds = total_image_ds.reshape((lunghezza, img_width, img_height, img_channel))  # resize
total_image_ds = image_ds.astype('float32') / 255                                   # normalization
total_labels_ds = to_categorical(total_labels_ds,num_classes=len(classes))          # transform label in categorical format

# -------- cross validation --------
folder_dim = len(total_image_ds) // k               # number of sample in each folder of the k-cross validation
print ("folder dim: ", folder_dim)

start_time = time.time()                            # start time for training
# slide along each pairs of parameters and does k-cross validation for each one of them
for batch in batch_size_list:       # slide the batch_size values
    batch_size = batch                  # update the actual 
    for early in early_patience:            # slide the early_patience
        # define the dictionary that will contain the results of the k-cross validation training
        name = "b"+str(batch)+"_e"+str(early)       # create the name for this combination of values parameters
        result_dict[name] = {}                      # define 
        result_dict[name]['loss'] = []              # define list of loss
        result_dict[name]['acc'] = []               # define list of accuracy
        
        for fold in range(k):               # do cross validation
            print("Pair: batch_size = ", batch," ,patience = ",early, " cicle ",fold," of ",k)
            network = make_model()                  # take the model
            
            # test data: data from partition k
            test_image = total_image_ds[fold * folder_dim: (fold + 1) * folder_dim]     # take img
            test_label = total_labels_ds[fold * folder_dim: (fold + 1) * folder_dim]    # take labels
            
            # training and validation data: data from all other partitions
            other_img = np.concatenate([total_image_ds[:fold * folder_dim],  total_image_ds[(fold+ 1) * folder_dim:]], axis=0)      # take other images
            other_label = np.concatenate([total_labels_ds[:fold * folder_dim],  total_labels_ds[(fold+ 1) * folder_dim:]], axis=0)  # take other labels
            # split in training and validtion set
            train_image, val_img, train_label, val_label = train_test_split(other_img, other_label, test_size=val_set_split , random_state=42, shuffle=True)
            
            del other_img                       # clear to free memory
            del other_label                     # clear to free memory
            
            # define early stopping
            checkpoint = ModelCheckpoint(filepath = path_check_point_model+'/weight_seg_'+name+".hdf5", verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max
            eStop = EarlyStopping(patience = early, verbose = 1, restore_best_weights = True, monitor='val_loss')
            
            # create TRAIN SET using generator function and specifying shapes and dtypes
            train_set = tf.data.Dataset.from_generator(generator_train, 
                                                     output_signature=(tf.TensorSpec(shape=(batch ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(batch, len(classes)), dtype=tf.float32)))
            
            # create VALIDATION SET using generator function and specifying shapes and dtypes
            val_set = tf.data.Dataset.from_generator(generator_val, 
                                                     output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32)))
            # create TEST SET using generator function and specifying shapes and dtypes
            test_set = tf.data.Dataset.from_generator(generator_test, 
                                                     output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32)))
            
            # train model
            history = network.fit(train_set,validation_data=val_set, epochs=epochs, callbacks = [checkpoint, eStop])     # fit model
            # evaluate
            loss_score , acc_score = network.evaluate(test_image, test_label)                      # obtain loss and accuracy metrics
            # save evaluate result
            result_dict[name]['loss'].append(loss_score)   
            result_dict[name]['acc'].append(acc_score) 
            
            del loss_score                      # clear to free memory
            del acc_score                       # clear to free memory
            
            # create the confusion matrix, rows indicate the real class and columns indicate the predicted class 
            conf_matrix = np.zeros((len(classes),len(classes)))     # at begin values are 0
            
            predictions = network.predict(test_image)               # get the output for each sample of the test set
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
                    conf_matrix_perc[i][j] = round( (conf_matrix[i][j]/len(test_image))*100 ,2)     # calculate percentage value
            
            # check if there is already a cinfusion matrix saved for this parameters
            if result_dict[name].get('conf_matrix') is None:
                # there isn't already a confusion matrix saved
                result_dict[name]['conf_matrix'] = conf_matrix                                      # assigne value
            else:
                # there is already a confusion matrix saved, sum new matrix with old metrix
                for i in range(conf_matrix.shape[0]):                   # rows
                    for j in range(conf_matrix.shape[1]):               # columns
                        result_dict[name]['conf_matrix'][i][j] += conf_matrix[i][j]                 # update value
                
            # check if there is already a cinfusion matrix percentages saved for this parameters
            if result_dict[name].get('conf_matrix_perc') is None:
                # there isn't already a confusion matrix percentages saved
                result_dict[name]['conf_matrix_perc'] = conf_matrix_perc                            # assigne value
            else:
                # there is already a confusion matrix percentages saved
                for i in range(conf_matrix.shape[0]):                   # rows
                    for j in range(conf_matrix.shape[1]):               # columns
                        result_dict[name]['conf_matrix_perc'][i][j] += conf_matrix_perc[i][j]       # update value
                    
            # clear variables to free memory
            back.clear_session()                            # clear session to free memory
            test_image = None
            test_label = None
            train_image = None
            val_img = None
            train_label = None
            val_label = None
            del predictions                                 # clear to free memory
            del network                                     # clear to free memory
            
        # print of the mean of result of the last k-cross validation done
        loss_scores = np.average(result_dict[name]['loss'])
        acc_scores = np.average(result_dict[name]['acc'])
        print("\n\nK-cross validation of the ", name ,"\nLoss mean :" ,loss_scores,"\nAccuracy mean : ",acc_scores ,"\n")
            
        # plot the confusion matrix
        rows = classes                                          # contain the label of the classes showed in the rowvalues of rows          
        columns = classes                                       # contain the label of the classes showed in the rowvalues of columns   

        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(columns)), labels=columns)
        ax.set_yticks(np.arange(len(rows)), labels=rows)
        
        for i in range(len(rows)):                              # rows
            for j in range(len(columns)):                       # columns
                # give the value in the confusion matrix
                conf_m_mean = result_dict[name]['conf_matrix'][i][j] // k                       # mean of confusion matrix in position i,j
                conf_m_perc_mean = round (result_dict[name]['conf_matrix_perc'][i][j] / k , 2)  # mean of confusion matrix percentages in position i,j
                value = str( str ( conf_m_mean ) + " (" + str( conf_m_perc_mean ) + "%)")       # concatenate the value of conf matrix and conf matrix percentage
                # assigne value of confusion matrix
                ax.text(x=j, y=i, s=value,
                               ha="center", va="center", size='x-large')
                
        plt.xlabel('Predictions', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        name_visualize = "Confusion Matrix ( " + "batch: "+str(batch)+" patience: "+str(early) + ")"
        plt.title(name_visualize, fontsize=14)
        plt.show()     
        
# -- make accuracy and loss mean value for each parameters pair
# this 3 list of the result to show have the same order
list_pair_param = list(result_dict.keys())                      # lsit containing all pairs of the parameters tested
list_acc = []                                                   # contain the mean value of the accuracy for each pairs of the parameters tested
list_loss = []                                                  # contain the mean value of the loss for each pairs of the parameters tested

# do the mean value for loss and accuracy per each pairs of parameters tested
for pair in list_pair_param:
    print("Prima della media\nresult of" + pair + " :" ,result_dict[pair]['loss'] , " : ",result_dict[pair]['acc'])
    # validation score: average of the validation scores of the k folds
    loss_scores = np.average(result_dict[pair]['loss'])
    acc_scores = np.average(result_dict[pair]['acc'])
    print("Dopo della media\nresult of" + name + " :" ,loss_scores," : ",acc_scores)
    list_acc.append(acc_scores)                                                
    list_loss.append(loss_scores)
    
# -- plot accuracy results
fig = plt.figure()
plt.plot(list_pair_param,list_acc ,'o-b')
plt.title(model_name)                       # plot title
plt.xlabel("Parameters")                    # x axis title
plt.ylabel("Accuracy")                      # y axis title
plt.show()                                  # show accuracy plot

# -- plot loss results
fig = plt.figure()
plt.plot(list_pair_param,list_loss ,'o-b')
plt.title(model_name)                       # plot title
plt.xlabel("Parameters")                    # x axis title
plt.ylabel("Loss")                          # y axis title
plt.show()                                  # show loss plot

end_time = time.time()                              # end time for training
print(f"Time to all cross validation of the model: {(end_time - start_time) // 60} (m)")  # print time to all cross validation of the model