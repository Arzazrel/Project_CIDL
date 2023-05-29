# -*- coding: utf-8 -*-
"""
@author: Made by Alessandro Diana

dataset used: https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images     All credit belongs to the respective owners and creators of the datasets
"""
# general
import os
import numpy as np
import random
# for model
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
# for image visualization
import matplotlib.pyplot as plt
import cv2
import PIL
from PIL import ImageTk
from PIL import Image
# for GUI
from tkinter import *
from  tkinter import ttk
# for thread
from threading import Thread
# for tensorflow-GPU
from tensorflow.python.client import device_lib 
# import of my files
#import Classifier as Clf

# ------------------------------------ start: global var ------------------------------------
# ---- GUI variables ----
window = Tk()
window_width = 800                              # is the width of the tkinter window
window_height = 700                             # is the height of the tkinter window
# explain frame
ex_f_padx = 10                                  # horizontal pad for the explain frame
ex_f_pady = 5                                   # vertical pad for the explain frame
ex_frame_width = window_width - 2*ex_f_padx     # width of the explain frame
ex_frame_height = 45                            # height of the explain frame
# top frame
t_f_padx = 10                                   # horizontal pad for the top frame
t_f_pady = 5                                    # vertical pad for the top frame
top_frame_width = window_width - 2*t_f_padx     # width of the top frame
top_frame_height = 140                          # height of the top frame
# image frame
im_f_padx = 10                                  # horizontal pad for the image frame
im_f_pady = 5                                   # vertical pad for the image frame
im_frame_width = window_width - 2*im_f_padx     # width of the image frame
im_frame_height = 250                           # height of the image frame
# bottom frame
b_f_padx = 10                                   # horizontal pad for the bottom frame
b_f_pady = 5                                    # vertical pad for the bottom frame
b_frame_width = window_width - 2*im_f_padx      # width of the bottom frame
b_frame_height = 140                            # height of the bottom frame
# error frame
er_f_padx = 10                                  # horizontal pad for the error frame
er_f_pady = 5                                   # vertical pad for the error frame
er_frame_width = window_width - 2*im_f_padx     # width of the error frame
er_frame_height = 45                            # height of the error frame

# ---- errors variables ----
error_text = StringVar()                        # text that shows the errors
error_text.set('')                              # default value: empty text
er_load_model_text = "Please insert a model name in the field before load CNN model."   # error text that occur when try to load a model without specifying the CNN name model
er_load_model_unknown_text = "There isn't a CNN model with the specified name."         # error text that occur when it's not possible return the specified model
er_save_model_text = "Please insert a model name in the field before save CNN model."   # error text that occur when try to save a model without specifying the CNN name model
er_no_ds_text = "Please load the dataset and retry."                                    # error text that occur when user want to work with dataset without loading one
er_train_without_ds_text = "Before train the model you must load image dataset."        # error text that occur when user want to make and fit the CNN model without loading dataset
er_eval_without_model_text = "Before evaluate the model you must make and fit or load a mode."  # error text that occur when user want to evaluate the model without make and fit the CNN model

# ---- status variables ----
model_trained = False                           # variable that show if there is a model trained
image_to_visualize = None                       # image that will be visualized in the GUI
index_image_visualized = -1
status_DS_text = StringVar()                    # text that shows the state of the dataset (missing, loading, loaded)
status_DS_text.set('Image DataSet: missing')    # the default value is 'missing'
CNN_menu_text = StringVar()                     # text that shows in the menu the type of CNN model select, the possible values are (None, AlexNet, GoogleNet). The chosen model can be train and fit
CNN_menu_text.set('None')                       # the default value is 'None'
status_model_text = StringVar()                 # text that shows the state of the CNN model (empty, trained)
status_model_text.set('CNN model: empty')       # the default value is 'empty'

# ---- label text variables ----
classify_text = StringVar()                     # text that shows the class of the new object classified by classifier
classify_text.set(':')                          # default value
label_image_text = StringVar()                  # text that shows the groundtruth of the visualised image
label_image_text.set('Label: ')                 # default value

# ---- path variables ----
path_dir_ds = "Dataset\Train_DS"                # folder in which there are the image ds for training
path_dir_test_ds = "Dataset\Test_DS"            # folder in which there are the image ds for testing
path_dir_model = "Model"
# ---- model variables ----
network = None                                  # contain the CNN model, default value is None
batch_size = 64                                 # batch size for training, this is the default value
img_height = 240                                # height of the images in input to CNN
img_width = 240                                 # width of the images in input to CNN
img_channel = 3                                 # channel of the images in input to CNN                
output_activation = 'softmax'                   # activation function of the output layer  
hidden_activation = 'relu'                      # activation function of the hidden layer
epochs = 100                                    # number of epochs for training, this is the deafault value

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
test_set_split = 0.2                # test set size as a percentage of the total dataset
val_set_split = 0.1                 # validation set size as a percentage of the training set

# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: methods for GUI ------------------------------------
# method that cleans GUI elements
def cleanGUI():
    list = window.grid_slaves()
    for l in list:
        l.destroy()

# method for handling the closing of the window by the user
def on_closing():
    # close window
    window.destroy()
    
# method executed for change the view visualised
def current_view_to_visualise():
    cleanGUI()                              # clean all element of the GUI
    # create variable for the GUI elements
    explainText = "Welcom to Ifrit.\nA fire Detection CNN."
    CNN_model_text = ['None','AlexNet','GoogleNet','Ifrit']

    # create the GUI elements and place them 
    # ---- start: explain_frame ----
    explain_frame = Frame(window, width=ex_frame_width , height=ex_frame_height , bg='grey')
    explain_frame.grid(row=0, column=0, padx=ex_f_padx , pady=ex_f_pady , sticky="nsew")
    explain_frame.grid_propagate(False)
    
    explainTextLabel = Label(explain_frame, text=explainText)               # Label to briefly explain
    explainTextLabel.grid(row=0, column=0, sticky="W", padx=window_width/2, pady=5)
    # ---- end: explain_frame ----
    
    # ---- start: top frame (contain: iport for dataset, import and save for model, select and fit model buttons) ----
    top_frame = Frame(window, width=top_frame_width , height=top_frame_height , bg='grey')
    top_frame.grid(row=1, column=0, padx=t_f_padx , pady=t_f_pady , sticky="nsew")
    top_frame.grid_propagate(False)
    
    # -- start: row 0 --
    btn_load_ds = Button(top_frame, text="Load image DS", command=btn_load_ds_method)   # button to load the whole dataset
    btn_load_ds.grid(row=0, column=0, sticky="W", padx=10, pady=10)
    
    dataset_label = Label(top_frame, textvariable=status_DS_text)           # label for the status of DS (missing,loading,loaded)
    dataset_label.grid(row=0, column=4, sticky="W", padx=10, pady=10)    
    # -- end: row 0 --
    
    # -- start: row 1 --
    name_model_label = Label(top_frame, text="CNN model name: ")                # label for the name of the CNN model to load or to save
    name_model_label.grid(row=1, column=0, sticky="W", padx=10, pady=10) 
    
    name_model_input = Entry(top_frame)                                     # entry for the CNN model name
    name_model_input.grid(row=1, column=1, sticky="WE", padx=10)
    
    btn_load_model = Button(top_frame, text="Load CNN model", command=lambda: load_saved_model(name_model_input.get()))   # button to load the CNN model
    btn_load_model.grid(row=1, column=2, sticky="W", padx=10, pady=10)
    
    btn_save_model = Button(top_frame, text="Save CNN model", command=lambda: save_model(name_model_input.get()))   # button to save the CNN model
    btn_save_model.grid(row=1, column=3, sticky="W", padx=10, pady=10)
    # -- end: row 1 --
    
    # -- start: row 2 --
    name_model_label = Label(top_frame, text="Select the CNN model that you want:") # label to explain the choice of CNN models
    name_model_label.grid(row=2, column=0, sticky="W", padx=10, pady=10) 
    
    CNN_menu = OptionMenu(top_frame, CNN_menu_text,*CNN_model_text)                 # creating select menu for CNN model
    CNN_menu.grid(row=2, column=1,padx=10)
    
    btn_fit_model = Button(top_frame, text="Fit CNN model", command=make_fit_model)      # button to fit CNN model
    btn_fit_model.grid(row=2, column=2, sticky="W", padx=10, pady=10)
    
    model_label = Label(top_frame, textvariable=status_model_text)           # label for the status of DS (missing,loading,loaded)
    model_label.grid(row=2, column=4, sticky="W", padx=10, pady=10)  
    # -- end: row 2 --
    # ---- end: top frame ----
    
    # ---- start: image frame ----
    image_frame = Frame(window, width=im_frame_width , height=im_frame_height , bg='grey')
    image_frame.grid(row=2, column=0, padx=im_f_padx , pady=im_f_pady , sticky="nsew")
    image_frame.grid_propagate(False)
    
    # image take by test set that will be predicted
    if image_to_visualize is not None:
        image_label = Label(image_frame, image= image_to_visualize)
        image_label.grid(row=0, column=1, sticky="W", padx=(window_width/2 - img_width/2), pady=5)
    # ---- end: image frame ----

    # ---- start: bottom frame (contains: button to perform load an image of the test set and predict) ----
    bottom_frame = Frame(window, width=b_frame_width , height=b_frame_height , bg='grey')
    bottom_frame.grid(row=3, column=0, padx=b_f_padx , pady=b_f_pady , sticky="nsew")
    bottom_frame.grid_propagate(False)
    
    # -- start: row 0 --
    btn_load_random_image = Button(bottom_frame, text="Take image", command=btn_load_image) # buttons to load a random image from test DS to predict 
    btn_load_random_image.grid(row=0, column=0, sticky="W", padx=10, pady=10)
    
    correct_label = Label(bottom_frame, textvariable=label_image_text)                      # label for the groundtruth
    correct_label.grid(row=0, column=1, sticky="W", padx=10, pady=10)
    
    btn_predict = Button(bottom_frame, text="Classify", command=predict)                    # button to predict label of image
    btn_predict.grid(row=0, column=2)
    
    result_classifier_label = Label(bottom_frame, textvariable=classify_text)               # label for the result of classifier
    result_classifier_label.grid(row=0, column=3, sticky="W", padx=0, pady=5)
    # -- end: row 0 --
    
    # -- start: row 1 --
    btn_load_test_ds = Button(bottom_frame, text="Load extern test DS", command=btn_load_ds_method)   # button to load the whole dataset
    btn_load_test_ds.grid(row=1, column=0, sticky="W", padx=10, pady=10)
    
    dataset_test_label = Label(bottom_frame, textvariable=status_DS_text)           # label for the status of DS (missing,loading,loaded)
    dataset_test_label.grid(row=1, column=4, sticky="W", padx=10, pady=10)
    # -- end: row 1 --
    
    # -- start: row 2 --
    btn_load_random_test_img = Button(bottom_frame, text="Take extern test image", command=btn_load_image) # buttons to load a random image from test DS to predict 
    btn_load_random_test_img.grid(row=2, column=0, sticky="W", padx=10, pady=10)
    
    correct_ext_test_label = Label(bottom_frame, textvariable=label_image_text)                      # label for the groundtruth
    correct_ext_test_label.grid(row=2, column=1, sticky="W", padx=10, pady=10)
    
    btn_predict = Button(bottom_frame, text="Classify", command=predict)                    # button to predict label of image
    btn_predict.grid(row=2, column=2)
    
    result_classifier_label = Label(bottom_frame, textvariable=classify_text)               # label for the result of classifier
    result_classifier_label.grid(row=2, column=3, sticky="W", padx=0, pady=5)
    # -- end: row 2 --
    # ---- end: bottom frame----
    
    # ---- start: error frame (contain the error text if occour an error) ----
    error_frame = Frame(window, width=er_frame_width , height=er_frame_height , bg='grey')
    error_frame.grid(row=4, column=0, padx=er_f_padx , pady=er_f_pady , sticky="nsew")
    error_frame.grid_propagate(False)
    
    error_label = Label(error_frame, textvariable=error_text, bg='grey')
    error_label.grid(row=0, column=0, padx=10, pady=10)
    # ---- end: error frame ----
    
# chose and take the image to visualize and predict, the image is chosen from test set
def btn_load_image():
    global image_to_visualize,index_image_visualized            # global variables references
    img = None                                                  # variable that contain the image to visualize
    index = 0                                                   # index of the chosen img on the test dataset
    error_text.set('')                                          # clean eventual text error
    
    if len(test_image) == 0 or len(test_label) == 0:            # check if there are images in test label
        if len(total_image_ds) != 0:                            # take image from total dataset
            print("prendo immaginde da total_ds")
            index = random.randint(0,len(total_image_ds)-1)     # chose a random index
            index_image_visualized = index      
            img = total_image_ds[index]                         # take random image from ds, the image at the index position
            label = str(classes[total_labels_ds[index]])        # take the label of the chosen image    
            label_image_text.set('Label: '+label)               # shows the label of the chosen image
        else:                                                   # no dataset loaded
            error_text.set(er_no_ds_text)                       # shows text error
    else:                                                       # take image from test dataset
        print("prendo immagine da set di test")
        index = random.randint(0,len(test_image)-1)             # chose a random index
        index_image_visualized = index
        img = test_image[index]*255                             # remember that the value of the image have been normalized
        label = str(classes[test_label[index]])                 # take the label of the chosen image 
        label_image_text.set('Label: '+label)                   # shows the label of the chosen image
    
    if img is not None:
        blue,green,red = cv2.split(img)                         # Rearrange colors
        img = cv2.merge((red,green,blue))
        im = PIL.Image.fromarray(np.uint8(img),'RGB')
        image_to_visualize = ImageTk.PhotoImage(image=im)       # update image to visualize in GUI
    
    current_view_to_visualise()                                 # update GUI

# ------------------------------------ end: methods for GUI ------------------------------------

# ------------------------------------ start: methods for DS ------------------------------------
# activate a thread to load the ds, in this way the GUI will not be blocked
def btn_load_ds_method():
    t = Thread(target=import_image_from_ds, args=(path_dir_ds,))
    t.start()
    
# method for import the whole dataset, path_ds is the path of the dataset to load. -- P.S. for more detail please read note 0 (at the end of the file)  
def import_image_from_ds(path_ds):
    global total_image_ds, total_labels_ds          #refer to global variables
    list_dir_ds = os.listdir(path_ds)               # list of the folders that are in the DS, one folder for each class
    
    status_DS_text.set('Image DataSet: loading')    # notify the start of the import
    # take the images and labels form DataSet
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
                    resize_img = cv2.resize(img, dim, interpolation= cv2.INTER_LINEAR)  # resize the image
                    total_image_ds.append(resize_img)                                   # add image to total_image_ds
                    total_labels_ds.append(index)                                       # add correlated label to total_lael_ds
                else:
                    total_image_ds.append(img)                                          # add image to total_image_ds
                    total_labels_ds.append(index)                                       # add correlated label to total_lael_ds
                    
    # convert in np.array
    total_image_ds = np.array(total_image_ds)
    total_labels_ds = np.array(total_labels_ds)
    # control data print
    print("Num of classes: ",len(classes))
    print("total_image_ds",len(total_image_ds), total_image_ds.shape)
    print("total_labels_ds",len(total_labels_ds), total_labels_ds.shape)
    print("Requied memory for images ds: ",total_image_ds.size * total_image_ds.itemsize / 10**9," GB")
    
    status_DS_text.set('Image DataSet: downloaded')             # notify the end of the process
    

# method for preprocessing and split the dataset
def make_set_ds():
    global test_image, test_label, train_image, train_label  # global variables references
    
    # ----- preprocessing and reshape ----
    lunghezza = len(total_image_ds)                                                     # take the len of the dataset
    image_ds = total_image_ds.reshape((lunghezza, img_width, img_height, img_channel))  # resize
    image_ds = image_ds.astype('float32') / 255                                         # normalization
    labels_ds = to_categorical(total_labels_ds,num_classes=len(classes))                # transform label in categorical format
    
    # ---- generete the trining and test set ----
    # split to generate train and test set
    train_img_temp, test_image, train_label_temp, test_label = train_test_split(image_ds, labels_ds, test_size=test_set_split , random_state=42, shuffle=True)
    # split to generate validation set from train set
    train_image, val_img, train_label, val_label = train_test_split(train_img_temp, train_label_temp, test_size=test_set_split , random_state=42, shuffle=True)

# ------------------------------------ end: methods for DS ------------------------------------

# ------------------------------------ start: methods for CNN model ------------------------------------
# method to create and fit model
def make_fit_model():
    global model_trained, test_image, test_label, train_image, train_label, network  # global variables references
    
    if len(total_image_ds) == 0:                        # control check
        error_text.set(er_train_without_ds_text)        # update error text
        return
    
    if len(test_image) == 0 or len(train_image) == 0:   # check whether the training and test set have already been made
        make_set_ds()                                       # split and create the sets
    
    error_text.set('')                                  # clear error text
    status_model_text.set('Model: working')             # notify the start of the process
    
    # ---- make the model -----
    if not model_trained:                       # check if there is a ready model or not 
        # check what type of model the user want to make and fit
        
        # Ifrit model
        network = models.Sequential()                                   # rete del modello
        network.add(layers.Conv2D(32, (3, 3),padding='same', activation=hidden_activation, input_shape=(img_width, img_height, img_channel)))
        network.add(layers.MaxPooling2D((3, 3)))
        network.add(layers.Conv2D(32, (3, 3), activation=hidden_activation))
        network.add(layers.MaxPooling2D((3, 3)))
        network.add(layers.Conv2D(32, (3, 3), activation=hidden_activation))
        network.add(layers.MaxPooling2D((3, 3)))
        network.add(layers.Flatten())
        network.add(layers.Dense(128, activation=hidden_activation))
        network.add(layers.Dropout(0.2))
        network.add(layers.Dense(128, activation=hidden_activation))
        network.add(layers.Dropout(0.2))
        network.add(layers.Dense(len(classes), activation=output_activation))

        # compile rmsprop o adam
        network.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    network.summary()                                   # summary of the CNN model
    
    # create TRAIN SET using generator function and specifying shapes and dtypes
    train_set = tf.data.Dataset.from_generator(generator_train, 
                                             output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32)))
    
    # create VALIDATION SET using generator function and specifying shapes and dtypes
    val_set = tf.data.Dataset.from_generator(generator_val, 
                                             output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32)))
    # create TEST SET using generator function and specifying shapes and dtypes
    test_set = tf.data.Dataset.from_generator(generator_test, 
                                             output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32)))
    
    
    history = network.fit(train_set,validation_data=val_set, epochs=epochs)     # fit model
    
    model_trained = True                                # update status variable
    status_model_text.set('Model: trained')             # notify the end of the process
    
    plot(history,"(Training set)")                      # visualize the value for the fit - history.history is a dictionary - call method for plot train result
    model_evaluate()                                    # evaluate the model 

# method for evaluate the model by the test set
def model_evaluate():
    error_text.set('')                                  # clear error text
    if not model_trained:                               # check control if there is a CNN model ready
        error_text.set(er_eval_without_model_text)      # update error text
        return
    
    # check if there are images in test label didn't use in the train, ( test_image , test_label)
    if len(test_image) == 0 or len(test_label) == 0:
        # split
        data_train, data_test, labels_train, labels_test = train_test_split(total_image_ds, total_labels_ds, test_size=test_set_split , random_state=42)
        lunghezza = len(data_test)                                                      # take the len of the dataset
        data_test = data_test.reshape((lunghezza, img_width, img_height, img_channel))  # resize
        data_test = data_test.astype('float32') / 255                                   # normalization
        labels_test = to_categorical(labels_test,num_classes=len(classes))              # transform label in categorical format
    else:
        data_test = test_image
        labels_test = test_label
    
    test_loss, test_acc = network.evaluate(data_test, labels_test)                      # obtain loss and accuracy metrics
    dict_metrics = {'loss': test_loss, 'accuracy': test_acc}                            # create a dictionary contain the metrics
    plot(dict_metrics,"(Test set)")                                                                  # plot the values obtained
    

# check if there is a saved model and load it
def load_saved_model(model_name):
    global network,model_trained                                # reference to a global variables
    error_text.set('')                                          # clean eventual text error
    if model_name:                                              # check if user has entered a model name
        save_path = os.path.join(path_dir_model,model_name)
        if os.path.exists(save_path):                           # check if there is a model
            network = load_model(save_path)                     # load model
            model_trained = True                                # update status variable
            status_model_text.set('Model: trained')
        else:
            error_text.set(er_load_model_unknown_text)          # shows text error
    else:
        error_text.set(er_load_model_text)                      # shows text error
        
# save model
def save_model(model_name):
    global network,model_trained                                # reference to a global variables
    error_text.set('')                                          # clean eventual text error
    if model_name:                                              # check if user has entered a model name
        save_path = os.path.join(path_dir_model,model_name)
        network.save(save_path)                                 # save the trained model, creates a HDF5 file
    else:
        error_text.set(er_save_model_text)                      # shows text error
        
# method to predict the label associated to the image visualized
def predict(): 
    if index_image_visualized != -1 and model_trained:      # control check
        error_text.set('')                                  # clean the error_text
        if len(test_image) == 0 or len(test_label) == 0:    # check to know what set is used (total_image_ds or total_test_image)
            # take from total_image_ds
            print("prendo immagine da set totale")
            img = total_image_ds[index_image_visualized].reshape((1, img_width, img_height, img_channel))
        else:
            # take from total_test_image
            print("prendo immagine da set di test")
            img = test_image[index_image_visualized].reshape((1, img_width, img_height, img_channel))
        # predict
        predictions = network.predict(img)                              # get the output for each sample
        classify_text.set(': '+str(classes[np.argmax(predictions)]))    # update GUI
    else:
        error_text.set('Before predict you must train model and load an image.')     # update text error
# ------------------------------------ end: methods for CNN model ------------------------------------

# ------------------------------------ start: utility method ------------------------------------
# method to plot accuracy and loss. arc is a dictionary with 'loss' and 'accuracy', explain_text is a text to explain better the title of the plot (e.g. (training set)) 
def plot(arc,explain_text):
    loss = arc.history["loss"]          # take loss values              
    acc = arc.history["accuracy"]       # take accuracy values
    # plot
    plt.figure()
    plt.plot(loss,'o-b')
    plt.title('Loss '+str(explain_text))
    plt.show()
    plt.figure()
    plt.plot(acc,'o-b')
    plt.title('Accuracy '+str(explain_text))
    plt.show()

# ------------------------------------ end: utility method ------------------------------------

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    window.title("Ifrit")
    window.geometry(str(window_width)+'x'+str(window_height))
    window.resizable(False, False)
    
    #set_background_image()          # set background image
    current_view_to_visualise()     # method for visualize the correct GUI
    
    # handle the window closing by the user
    #window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()
    
# -------- Notes --------
# -- Note 0 --
# in "import_image_from_ds" method I chose an approch more flexible than necessary. Array classes will contain the labels of the classes,
# in this way the classes number and names are not fixed apriori but it's calculated at real time. for this fire detection problem 
# is not necessary but in this way the code and the app are more flexible and usable for future improve and usage.

# -- Note 1 --
