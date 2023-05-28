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
ex_f_pady = 10                                  # vertical pad for the explain frame
ex_frame_width = window_width - 2*ex_f_padx      # width of the explain frame
ex_frame_height = 60                           # height of the explain frame
# top frame
t_f_padx = 10                                   # horizontal pad for the top frame
t_f_pady = 10                                   # vertical pad for the top frame
top_frame_width = window_width - 2*t_f_padx     # width of the top frame
top_frame_height = 200                          # height of the top frame
# image frame
im_f_padx = 10                                  # horizontal pad for the image frame
im_f_pady = 10                                  # vertical pad for the image frame
im_frame_width = window_width - 2*im_f_padx     # width of the image frame
im_frame_height = 250                           # height of the image frame
# error frame
er_f_padx = 10                                  # horizontal pad for the error frame
er_f_pady = 10                                  # vertical pad for the error frame
er_frame_width = window_width - 2*im_f_padx     # width of the error frame
er_frame_height = 100                           # height of the error frame

# ---- errors variables ----
error_text = StringVar()                        # text that shows the errors
error_text.set('')                              # default value: empty text
er_load_model_text = "Please insert a model name in the field before load CNN model."   # error text that occur when try to load a model without specifying the CNN name model
er_load_model_unknown_text = "There isn't a CNN model with the specified name."         # error text that occur when it's not possible return the specified model
er_save_model_text = "Please insert a model name in the field before save CNN model."   # error text that occur when try to save a model without specifying the CNN name model

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
# ---- path variables ----
path_dir_ds = "Dataset\Train_DS"                # folder in which there are the image ds for training
path_dir_test_ds = "Dataset\Test_DS"            # folder in which there are the image ds for testing
path_dir_model = "Model"
# ---- model variables ----
network = None #models.Sequential()                  # contain the CNN model, default value is None
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
    explainTextLabel.grid(row=0, column=0, sticky="W", padx=20, pady=10)
    # ---- end: explain_frame ----
    
    # ---- start: top frame (contain: iport for dataset, import and save for model, select and fit model buttons) ----
    top_frame = Frame(window, width=top_frame_width , height=top_frame_height , bg='grey')
    top_frame.grid(row=1, column=0, padx=t_f_padx , pady=t_f_pady , sticky="nsew")
    top_frame.grid_propagate(False)
    
    # -- start: row 0 --
    btn_load_ds = Button(top_frame, text="Load image DS", command=btn_load_ds_method)   # button to load the whole dataset
    btn_load_ds.grid(row=0, column=0, sticky="W", padx=10, pady=10)
    
    dataset_label = Label(top_frame, textvariable=status_DS_text)           # label for the status of DS (missing,loading,loaded)
    dataset_label.grid(row=0, column=1, sticky="W", padx=10, pady=10)    
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
    name_model_label = Label(top_frame, text="Select the CNN model that you want:")      # label to explain the choice of CNN models
    name_model_label.grid(row=2, column=0, sticky="W", padx=10, pady=10) 
    
    CNN_menu = OptionMenu(top_frame, CNN_menu_text,*CNN_model_text)                      # creating select menu for CNN model
    CNN_menu.grid(row=2, column=1,padx=10)
    
    btn_fit_model = Button(top_frame, text="Fit CNN model", command=btn_load_ds_method)  # button to fit CNN model
    btn_fit_model.grid(row=2, column=2, sticky="W", padx=10, pady=10)
    # -- end: row 2 --
    # ---- end: top frame ----
    
    # ---- start: image frame ----
    image_frame = Frame(window, width=im_frame_width , height=im_frame_height , bg='grey')
    image_frame.grid(row=2, column=0, padx=im_f_padx , pady=im_f_pady , sticky="nsew")
    image_frame.grid_propagate(False)
    
    # image take by test set that will be predicted
    if image_to_visualize is not None:
        image_label = Label(image_frame, image= image_to_visualize)
        image_label.grid(row=3, column=1, sticky="W", padx=10, pady=10)
    # ---- end: image frame ----

    # ---- start: ----
    # ---- end: ----
    
    # ---- start: error frame (contain the error text if occour an error) ----
    error_frame = Frame(window, width=er_frame_width , height=er_frame_height , bg='grey')
    error_frame.grid(row=2, column=0, padx=er_f_padx , pady=er_f_pady , sticky="nsew")
    error_frame.grid_propagate(False)
    
    error_label = Label(error_frame, textvariable=error_text, bg='grey')
    error_label.grid(row=0, column=0, padx=10, pady=10)
    # ---- end: error frame ----
    
    
    
    """
    # create variable for the GUI elements
    explainText = "Welcome to Pokemon Image Classifier.\n"
    # create the GUI elements and place them 
    # main frame
    main_frame = Frame(window, width=760, height=560, bg='grey')
    main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
    main_frame.grid_propagate(False)
    
    # -- inizio - riga 0 --
    # label for explain text
    explainTextLabel = Label(main_frame, text=explainText)
    explainTextLabel.grid(row=0, column=1, sticky="W", padx=35, pady=20)
    # -- fine - riga 0 --
    
    # -- inizio - riga 1 --
    # buttons to load all the dataset
    btn_load_ds = Button(main_frame, text="Load image dataset", command=btn_load_ds_method)
    btn_load_ds.grid(row=1, column=0, sticky="W", padx=10, pady=10)
    
    # buttons to train and fit DNN
    btn_make_DNN = Button(main_frame, text="Train and fit DNN", command=make_model)
    btn_make_DNN.grid(row=1, column=1, sticky="W", padx=10, pady=10)
    
    # label for the result of classifier
    dataset_label = Label(main_frame, textvariable=status_DS_text)
    dataset_label.grid(row=1, column=2, sticky="W", padx=10, pady=10)
    
    # label for the result of classifier
    model_label = Label(main_frame, textvariable=status_DNN_text)
    model_label.grid(row=1, column=3, sticky="W", padx=10, pady=10)
    # -- fine - riga 1 --
    
    # -- inizio - riga 2 --
    # button to load the test image dataset
    btn_import_test_image = Button(main_frame, text="Load test image dataset:", command=btn_load_test_ds_method)
    btn_import_test_image.grid(row=2, column=0, sticky="W", padx=10, pady=10)
    
    # label for the result of classifier
    btn_import_test_image_label = Label(main_frame, textvariable=test_image_label)
    btn_import_test_image_label.grid(row=2, column=1, sticky="W", padx=10, pady=10)
    # -- fine - riga 2 --
    
    # -- inizio - riga 3 --
    #image to predict in the center
    #Create a Label to display the image
    if image_to_visualize is not None:
        image_label = Label(main_frame, image= image_to_visualize)
        image_label.grid(row=3, column=1, sticky="W", padx=10, pady=10)
    # -- fine - riga 3 --
    
    # -- inizio - riga 4 --
    # buttons to load a radom image from DS to predict 
    btn_load_random_image = Button(main_frame, text="Load image", command=btn_load_image)
    btn_load_random_image.grid(row=4, column=0, sticky="W", padx=10, pady=10)
    
    # label for the groundtruth
    correct_label = Label(main_frame, textvariable=label_image_text)
    correct_label.grid(row=4, column=1, sticky="W", padx=10, pady=10)
    
    # button to predict label of image
    btn_predict = Button(main_frame, text="Classify", command=predict)
    btn_predict.grid(row=4, column=2)
    
    # label for the result of classifier
    result_classifier_label = Label(main_frame, textvariable=classify_text)
    result_classifier_label.grid(row=4, column=3, sticky="W", padx=0, pady=5)
    # -- fine - riga 4 --
    
    # -- inizio - riga 5 --
    # buttons to load a radom image from DS to predict 
    btn_load_random_test_image = Button(main_frame, text="Load test image", command=load_image_test)
    btn_load_random_test_image.grid(row=5, column=0, sticky="W", padx=10, pady=10)
    
    # label for the groundtruth
    correct_test_label = Label(main_frame, textvariable=label_image_text)
    correct_test_label.grid(row=5, column=1, sticky="W", padx=10, pady=10)
    # -- fine - riga 5 --
    
    # -- inizio - riga 6 --
    # label for the error text
    error_text_label = Label(main_frame, textvariable=error_text)
    error_text_label.grid(row=6, column=1, sticky="W", padx=0, pady=5)
    # -- fine - riga 6 --
    """

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
        p = os.path.join(path_ds,folder)        # path of each folder
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
    

# ------------------------------------ end: methods for DS ------------------------------------

# ------------------------------------ start: methods for CNN model ------------------------------------
# check if there is a saved model and load it
def load_saved_model(model_name):
    global network,model_trained                                # reference to a global variables
    error_text.set('')                                          # clean eventual text error
    if model_name:                                              # check if user has entered a model name
        save_path = os.path.join(path_dir_model,model_name)
        if os.path.exists(save_path):                           # check if there is a model
            network = load_model(save_path)                     # load model
            model_trained = True                                # update status variable
            status_DNN_text.set('Model: trained')
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
        
# ------------------------------------ end: methods for CNN model ------------------------------------

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    window.title("Ifrit")
    window.geometry(str(window_width)+'x'+str(window_height))
    window.resizable(False, False)
    
    #set_background_image()          # set background image
    #load_saved_model()              # method for load a model if is already exist
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
