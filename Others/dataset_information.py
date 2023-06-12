# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

Explanation: 
    program to see properties and characteristics of the dataset such as the number of classes, how many images per class, 
    and general information on the size of the images
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------ start: global var ------------------------------------
# ---- status variable ----
del_corrupt_img = False                         # variable that indicate to the program whether it should delete any corrupted images in the dataset

# ---- dataset variables ----
img_number = 0                                  # total number of images in the dataset
classes = {}                                    # dictionary containing all the classes in the dataset and the number of images of each class
format_dict = {}                                # dictionary containing all the image formats in the dataset and for each of them the number of images
shape_dict = {}                                 # dictionary containing all the image shapes in the dataset and for each of them the number of images
top_shape_images = 10                           # the top frequent shapes for the images in the dataset
corrupt_images = []                             # array that will contain the name of the corrupted images that are in the dataset
RGB_mean_dict = {}                              # contain for each class the value for the color Blue , Green, Red. For a quick comparison of the colour distribution in the various classes and dataset.
                                                # the dictionary will have a key for each class and the corresponding value will be an array of three values.
# ---- path variables ----
path_dir_ds = "Dataset\Train_DS"                # folder in which there are the image ds for training
path_ds = os.path.join(os.pardir,path_dir_ds)   # complete folder to reach the ds -- P.S. For more detail read note 0, please (at the end of the file) 
list_dir_ds = os.listdir(path_ds)               # list of the folders that are in the DS, one folder for each class
# ------------------------------------ end: global var ------------------------------------

# slide the images and labels form DataSet
for folder in list_dir_ds:                      # for each folder in DS
    if classes.get(str(folder)) is None:        # check if the classes is already registered
        classes[str(folder)] = 0                # set counter equal 0
    
    p = os.path.join(path_ds,folder)        # path of each folder
    
    for filename in os.listdir(p):                      # for each images on the current folder
        img_number +=1                                  # update images counter
        classes[str(folder)] += 1                       # update images in classes counter
        # I get the image format, I assume that there are no dots in the name of the images
        split_name = str(filename).split('.')           # split the filename
        format_name = split_name[-1]                    # get the format
        # check format
        if format_dict.get(format_name) is None:     # check if the format is already registered
            format_dict[format_name] = 1             # set counter equal 0
        else:
            format_dict[format_name] += 1            # update counter    
        
        img = cv2.imread(os.path.join(p,filename))      # take current iamge
        if img is not None:                             # check image taken
            # check shape
            if shape_dict.get(str(img.shape)) is None:     # check if the shape is already registered
                shape_dict[str(img.shape)] = 1             # set counter equal 0
            else:
                shape_dict[str(img.shape)] += 1            # update counter  
            # see the color contribution
            # for each image calculate the mean of the value of the Red, Green and Blue
            red_mean = img[:,:,2].mean()
            green_mean = img[:,:,1].mean()
            blue_mean = img[:,:,0].mean()
            if RGB_mean_dict.get(str(folder)) is None:                              # check if there are already value for this class
                RGB_mean_dict[str(folder)] = [blue_mean,green_mean,red_mean]        # set with value
            else:
                temp_RGB = RGB_mean_dict[str(folder)]
                new_RGB = [temp_RGB[0] + blue_mean, temp_RGB[1] + green_mean , temp_RGB[2] + red_mean ]
                RGB_mean_dict[str(folder)] = new_RGB                                # update value
        else:                                           # corrupted image
            corrupt_images.append(str(filename))        # update 
            if del_corrupt_img:                         # check if the porgram has to delete the corrupted images or not
                os.remove(os.path.join(p,filename))     # delete the images

# Dataset characteristics output
print("The dataset has " , img_number, " images.")                                      # total number of images in dataset
print("The number of classes are ",len(classes.keys()), ". ", list(classes.keys()))     # number of classes and name of each class
print("List of the classes and number of related images for each one")
sorted_class_dict = dict(sorted(classes.items(), key=lambda x: x[1], reverse=True))     # sort the classes dictionary. reverse = true the biggest class will be the first                                                                            # initialize the counter
for k,v in sorted_class_dict.items():                                                   # for to slide the sorted dict
    print(k, ': ', v)                                                                   # print key and value       

if len(corrupt_images) != 0:                                                            #check if there are corrupted images
    print("There are ", len(corrupt_images)," corrupted images.\n",corrupt_images)

# number of shapes and images number for each shape
print("The number of different images shapes are ",len(shape_dict.keys()))
#for k,v in shape_dict.items():                                                          # for to see all images shapes
#    print("Shape: ",k," number of images: ",v)
# for to see the top k shape image with k as top_shape_images
sorted_shape_dict = dict(sorted(shape_dict.items(), key=lambda x: x[1], reverse=True))  # sort the shape_dictionary. reverse = true the most used shape will be the first
count = 0                                                                               # initialize the counter
for k,v in sorted_shape_dict.items():                                                   # for to slide the sorted dict
    if count >= top_shape_images:                                                       # check if is out of band for top k
        break
    print(k, ': ', v)                                                                   # print key and value
    count +=1                                                                           # update counter   

# number of format and images number for each format
print("The number of different images format are ",len(format_dict.keys()))
#for k,v in format_dict.items():                                                         # for to see all images format
#    print("Format: ",k," number of images: ",v)
# for to see the top k format image with k as top_shape_images
sorted_format_dict = dict(sorted(format_dict.items(), key=lambda x: x[1], reverse=True))  # sort the shape_dictionary. reverse = true the most used shape will be the first
count = 0                                                                               # initialize the counter
for k,v in sorted_format_dict.items():                                                  # for to slide the sorted dict
    if count >= top_shape_images:                                                       # check if is out of band for top k
        break
    print(k, ': ', v)                                                                   # print key and value
    count +=1                                                                           # update counter  
         
# show a plot that displays BRG bar gram for each class
for k,v in RGB_mean_dict.items():                                   # for to slide the sorted dict
    color = ["Blue", "Green", "Red"]                                # laber for bar in the bar gram                        
    x_pos = np.arange(len(color))
    plt.bar(x_pos, v, align='center')
    plt.xticks(x_pos, color)
    plt.ylabel('Value')
    plt.xlabel('Color')
    plt.title(k)                                                    # the title of the plot is the class 
    plt.show()                                                      # shows bar grams

"""
-------- Notes --------
-- Note 0 --
The path to the dataset depends on the location of the dataset and this file. 
The path variables work as long as you keep the structure of the project unchanged, 
if you change the structure or the dataset to be examined you update the paths for a correct program functioning.
"""


