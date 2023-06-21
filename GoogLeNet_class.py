# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that reproduces the GoogLeNet model

description: network description at the end of the file
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
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

# class that implement the GoogLeNet model
class GoogLeNet:
    
    # constructor
    def __init__(self,class_number):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = 224                   # height of the images in input to CNN
        self.img_width = 224                    # width of the images in input to CNN
        self.img_channel = 3                    # channel of the images in input to CNN (RGB)
        
        
    # method for make the model of the CNN. Due to the structure of the network (inception and parallel paths) it is better to use keras Models class instead of sequential
    def make_model(self):
        
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
        
        # first auxiliary classifier
        aux_1 = layers.AveragePooling2D((5, 5), strides=3)(seq_0)
        aux_1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux_1)
        aux_1 = layers.Flatten()(aux_1)
        aux_1 = layers.Dense(1024, activation='relu')(aux_1)
        aux_1 = layers.Dropout(0.7)(aux_1)
        aux_1 = layers.Dense(self.num_classes, activation='softmax')(aux_1)         # aux output layer
        
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
        aux_2 = layers.Dense(self.num_classes, activation='softmax')(aux_2)         # aux output layer
        
        # seq_2: is the last part of the CNN network starting with the end of seq_1 and ending with the end of CNN
        seq_2 = inception_mod(seq_1, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
        seq_2 = layers.MaxPooling2D(3, strides=2)(seq_2)
        seq_2 = inception_mod(seq_2, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
        seq_2 = inception_mod(seq_2, fil_1x1=384, fil_1x1_3x3=192, fil_3x3=384, fil_1x1_5x5=48, fil_5x5=128, fil_m_pool=128)
        seq_2 = layers.GlobalAveragePooling2D()(seq_2)
        seq_2 = layers.Dropout(0.4)(seq_2)
        out = layers.Dense(self.num_classes, activation='softmax')(seq_2)           # output layer
        
        self.model = Model(inputs = inp, outputs = [out, aux_1, aux_2])             # assign the CNN in model
        self.model.summary()
                
    # method for compile the model
    def compile_model(self):
        self.model.compile(optimizer='adam', 
              loss=[losses.sparse_categorical_crossentropy,
                    losses.sparse_categorical_crossentropy,
                    losses.sparse_categorical_crossentropy],
              loss_weights=[1, 0.3, 0.3],
              metrics=['accuracy'])

    # methd for take the training and validation set
    def take_ds(self,t_image, t_labels, v_image, v_labels,truncate,batch_size):
        # var ds
        self.train_image = t_image          # var that contain images of training set
        self.train_labels = t_labels        # var that contain labels of training set
        self.val_image = v_image            # var that contain images of validation set
        self.val_labels = v_labels          # var that contain labels of validation set
        # var for truncate or not the sets
        self.truncate_set = truncate
        self.batch_size = batch_size

        #print("train image: ",len(self.train_image), " train labels: ",len(self.train_labels))
        #print("val image: ",len(self.val_image), " val labels: ",len(self.val_labels))

    # method for fit the model (return history)
    def fit_model(self,epoch):
        # call the generator
        # create TRAIN SET using generator function and specifying shapes and dtypes
        t_img = tf.data.Dataset.from_generator(self.gen_train_image, 
                                                 output_signature=(tf.TensorSpec(shape=(self.batch_size ,self.img_width , self.img_height , self.img_channel), dtype=tf.float32)))

        t_lab = tf.data.Dataset.from_generator(self.gen_train_labels, 
                                                 output_signature=(tf.TensorSpec(shape=(self.batch_size, self.num_classes), dtype=tf.float32)))
        
        t_train = tf.data.Dataset.from_generator(self.train_generator, 
                                                 output_signature=(tf.TensorSpec(shape=(self.batch_size ,self.img_width , self.img_height , self.img_channel), dtype=tf.float32),
                                                                   (tf.TensorSpec(shape=(self.batch_size, self.num_classes), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(self.batch_size, self.num_classes), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(self.batch_size, self.num_classes), dtype=tf.float32))))

        t_train_1 = tf.data.Dataset.from_generator(self.train_generator, 
                                                 output_signature=(tf.TensorSpec(shape=(self.batch_size ,self.img_width , self.img_height , self.img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(3,self.batch_size,self.num_classes), dtype=tf.int32)))
        t_train_1 = tf.data.Dataset.from_generator(self.train_generator,
                                                   output_signature=(tf.TensorSpec(shape=(self.batch_size ,self.img_width , self.img_height , self.img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(32,), dtype=tf.int32)))
        t_train_2 = tf.data.Dataset.from_generator(self.train_generator, 
                                                 output_signature=(tf.TensorSpec(shape=(self.batch_size ,self.img_width , self.img_height , self.img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(32, 3,), dtype=tf.float32)))

        v_img = self.gen_val_image()
        v_lab = np.array(self.train_labels)
        #mult_v_lab = np.array([self.train_labels,self.train_labels,self.train_labels])
        #print("Shape di una labels: ", v_lab.shape, "Shape di multi labels: ", mult_v_lab.shape)
        # fit the model
        history = self.model.fit((t_img, t_lab), batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit((t_img, [t_lab,t_lab,t_lab]), batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit(t_train, batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit(t_train_1, batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit(t_train_2, batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit(self.train_image,(self.train_labels,self.train_labels,self.train_labels), batch_size=self.batch_size, epochs=epoch)
        #history = self.model.fit(self.train_image,self.train_labels, batch_size=self.batch_size, epochs=epoch)
        #validation_data=(v_img, (v_lab, v_lab, v_lab))
        #print("Item di history", history.items())
        # 
        
    # method for return the model
    def return_model(self):
        return self.model

    # ------------------------------------ start: generator ds methods ------------------------------------
    def train_generator_1(self):
        # create the tensor that will contain the data
        img_tensor = []                                             # tensor that contain the images of one batch from the set
        label_tensor = []                                           # tensor that contain the labels of one batch from the set
        img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
        label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.train_image) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.train_image)):                         # organize the sample in batch
            # take one image and the corresponding labels
            img = self.train_image[idx]                              
            label = self.train_labels[idx]
            # add new element and convert to TF tensors
            img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_tensor.append(tf.convert_to_tensor([label[0],label[1],label[0],label[1],label[0],label[1]], dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
                label_rest_tensor.append(tf.convert_to_tensor([label[0],label[1],label[0],label[1],label[0],label[1]], dtype=tf.float32))

            if len(img_tensor) == self.batch_size:                       # check to see if batch is full (reached batch_size)
                yield img_tensor, label_tensor                      # return the batch
                # clean list
                img_tensor.clear()
                label_tensor.clear()
            
            if idx == (len(self.train_image) - 1):                       # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                              # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        img_tensor.append(img_rest_tensor[i])
                        label_tensor.append(label_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield img_tensor, label_tensor              # return the last batch
    
    def train_generator(self):
        # create the tensor that will contain the data
        img_tensor = []                                             # tensor that contain the images of one batch from the set
        label_tensor = []                                           # tensor that contain the labels of one batch from the set
        img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
        label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.train_image) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.train_image)):                         # organize the sample in batch
            # take one image and the corresponding labels
            img = self.train_image[idx]                              
            label = self.train_labels[idx]
            # add new element and convert to TF tensors
            img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
                label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

            if len(img_tensor) == self.batch_size:                       # check to see if batch is full (reached batch_size)
                yield img_tensor, np.array([label_tensor,label_tensor,label_tensor])                      # return the batch
                # clean list
                img_tensor.clear()
                label_tensor.clear()
            
            if idx == (len(self.train_image) - 1):                       # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                              # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        img_tensor.append(img_rest_tensor[i])
                        label_tensor.append(label_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield img_tensor, np.array([label_tensor,label_tensor,label_tensor])              # return the last batch

    # define generator function to do the training set images
    def gen_train_image(self):
        # create the tensor that will contain the data
        img_tensor = []                                             # tensor that contain the images of one batch from the set
        img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.train_image) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.train_image)):                         # organize the sample in batch
            # take one image
            img = self.train_image[idx]                              
            # add new element and convert to TF tensors
            img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))

            if len(img_tensor) == self.batch_size:          # check to see if batch is full (reached batch_size)
                yield img_tensor                            # return the batch
                img_tensor.clear()                          # clean list
            
            if idx == (len(self.train_image) - 1):                  # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                         # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        img_tensor.append(img_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield img_tensor                                # return the last batch

    # define generator function to do the training set labels
    def gen_train_labels(self):
        # create the tensor that will contain the data
        label_tensor = []                                           # tensor that contain the labels of one batch from the set
        label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.train_labels) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.train_labels)):                        # organize the sample in batch
            # take one image and the corresponding labels]                              
            label = self.train_labels[idx]
            # add new element and convert to TF tensors
            label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

            if len(img_tensor) == self.batch_size:                       # check to see if batch is full (reached batch_size)
                yield label_tensor                      # return the batch
                label_tensor.clear()                    # clean list
            
            if idx == (len(self.train_labels) - 1):                      # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                              # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        label_tensor.append(label_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield label_tensor                 # return the last batch

    # define generator function to do the validation set images
    def gen_val_image(self):
        # create the tensor that will contain the data
        img_tensor = []                                             # tensor that contain the images of one batch from the set
        img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.val_image) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.val_image)):                         # organize the sample in batch
            # take one image
            img = self.val_image[idx]                              
            # add new element and convert to TF tensors
            img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))

            if len(img_tensor) == self.batch_size:          # check to see if batch is full (reached batch_size)
                yield img_tensor                            # return the batch
                img_tensor.clear()                          # clean list
            
            if idx == (len(self.val_image) - 1):                  # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                         # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        img_tensor.append(img_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield img_tensor                                # return the last batch

    # define generator function to do the validation set labels
    def gen_val_labels(self):
        # create the tensor that will contain the data
        label_tensor = []                                           # tensor that contain the labels of one batch from the set
        label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
        if not self.truncate_set:                                        # check if it has to truncate or not the set
            rest = self.batch_size - (len(self.val_labels) % self.batch_size)     # check if the division by batch_size produce rest
            #print("Training test rest: ",rest)
        else:
            rest = self.batch_size                                       # set always truncated
        #print("lunghezza totale: ", len(total_train_image), " Batch_size: ",batch_size, " modulo: ",len(total_train_image) % batch_size, " mancante(rest): ",rest)
        for idx in range(len(self.val_labels)):                        # organize the sample in batch
            # take one image and the corresponding labels]                              
            label = self.val_labels[idx]
            # add new element and convert to TF tensors
            label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
            if rest != self.batch_size and idx < rest:                   #check for the rest
                #print("aggiungo elemento ",idx," al contenitore per riempire il batch size finale")
                # add this sample for the future (sample in the rest)
                label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

            if len(img_tensor) == self.batch_size:                       # check to see if batch is full (reached batch_size)
                yield label_tensor                      # return the batch
                label_tensor.clear()                    # clean list
            
            if idx == (len(self.val_labels) - 1):                      # check if the set is finished, last batch
                #print("arrivato alla fine")
                if rest != self.batch_size:                              # check if there are rest to fix
                    #print("Sono in ultimo batch\nLunghezza ad ora del batch: ",len(img_tensor))
                    #there are samples that don't complete a batch, add rest sample to complete the last batch
                    for i in range(rest):
                        label_tensor.append(label_rest_tensor[i])
                    #print("Generator training set, arrivato a posizione: ",idx)
                    yield label_tensor                  # return the last batch
    # ------------------------------------ end: generator ds methods ------------------------------------


    
"""
brief description:
    GoogLeNet won ILSVRC-2014 and is one of the most successful models of the earlier years of CNN.
    GoogLeNet was created by Google Inc. and the model was published in the paper "Going Deeper with Convolutions".
    It uses different types of methods that allow it to create a deeper architecture. 
    
    1×1 convolution: These convolutions are used to decrease the number of parameters (weights and biases) of the architecture and increase the depth of the architecture. 
    Global Average Pooling: Fully connected layers contain most of the parameters of many architectures, which causes an increase in computing cost. 
                            In the GoogLeNet architecture, a method called global average pooling is used at the end of the network. This level takes a feature map of 7×7 and the average of 1×1. 
                            This reduces the number of trainable parameters to 0 and improves accuracy.
    Inception module: In this module the 1×1, 3×3, 5×5 convolution and 3×3 max pooling are executed in parallel to the input and the output of these is stacked together to generate the final output. 
                      The idea behind this is that convolution filters of different sizes handle multiple scale objects better.
    The overall architecture has a depth of 22 layers. The architecture also contains two auxiliary classifier layers connected to the output via inception layers.
    This architecture takes images of size 224 x 224 with RGB colour channels. All convolutions in this architecture use rectified linear units (ReLUs) as activation functions.

The architectural details of the auxiliary classifiers are as follows:
- A medium pooling layer with filter size 5×5 and step 3.
- A 1×1 convolution with 128 filters for size reduction and ReLU activation.
- A fully connected layer with 1025 outputs and ReLU activation.
- Dropout adjustment with dropout ratio = 0.7
- A softmax classifier with 1000 output classes similar to the main softmax classifier.
    
"""