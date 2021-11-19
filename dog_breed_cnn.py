import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import image
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import pdb

warnings.simplefilter(action="ignore", category=FutureWarning)


# # GPU support
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# model_test = Sequential({
#     Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu", input=INSERTDOG),
#     Conv2D(filters=32, kernel_size=(3,3), padding="same",activation="relu")
#     Conv2D(filters=64, kernel_size=(3,3), padding="same",activation="relu")
#     Conv2D(filters=128, kernel_size=(3,3), padding="same",activation="relu")
# })

# Image processing

ROOT="d:/csc2280/dog_breed_project"

training_path = "d:/csc2280/dog_breed_project/dog-breed-identification/train"
test_path = "d:/csc2280/dog_breed_project/dog-breed-identification/test_subset"   #this actually goes the the test_subset folder, which contains some of the dog pictures from test
validation_path = "d:/csc2280/dog_breed_project/dog-breed-identification/validation"

# for c in random.sample(glob.glob("d:/csc2280/dog_breed_project/dog-breed-identification/test/*"),1000):
#     shutil.move(c,test_path)
# for c in random.sample(glob.glob("d:/csc2280/dog_breed_project/dog-breed-identification/test/*"),100):
#     shutil.move(c,validation_path)
# train_test_split function in keras?
# validation_split

# actually changed these to the d: drive

# os.chdir("c:/users/vassr/dog-breed-identification")
# if os.path.isdir("test/test") is False:
#     os.makedirs("test/test")
#     os.makedirs("test/valid")
# os.chdir("../../")  #Not sure what this part is supposed to do

# dictionary containing dog breeds and the codes for the images that correspond to them
breed_dictionary ={}
label_dict = open(os.path.join(ROOT,"dog-breed-identification","dog_breed_labels_ids.txt"),"r")
for line in label_dict:
    line = line.split()
    breed_id = line[0]
    breed_name = line[1]
    breed_dictionary.setdefault(breed_name,[]).append(breed_id)
label_dict.close()

breed_list= list(breed_dictionary.keys())

# These three if statements produce folders for each breed of dog in the training, test, and validation directories
if os.path.isdir(os.path.join(training_path,"affenpinscher")) is False:
    for i in range(len(breed_dictionary.keys())):
            os.makedirs(os.path.join(training_path, breed_list[i]))

if os.path.isdir(os.path.join(test_path,"affenpinscher")) is False:
    for i in range(len(breed_dictionary.keys())):
            os.makedirs(os.path.join(test_path, breed_list[i]))

if os.path.isdir(os.path.join(validation_path,"affenpinscher")) is False:
    for i in range(len(breed_dictionary.keys())):
            os.makedirs(os.path.join(validation_path, breed_list[i]))


# attempting to add correct dog breeds to appropriate breed folder in the training folder
if os.path.isfile(os.path.join(training_path,"boston_bull","000bec180eb18c7604dcecc8fe0dba07")) is False:
    for breed in breed_list:
        for j in os.path.join(training_path,breed_dictionary[breed]):
            shutil.move(j,os.path.join(training_path,breed))


training_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory = training_path,target_size=(224,224), classes=breed_list#make 1 folder for each breed with name of breed being the folder name
)

# pdb.set_trace()

test_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory = test_path,target_size=(224,224),classes=breed_list)


valid_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory = validation_path,target_size=(224,224),classes=breed_list
)

# Making the vgg16 model and freezing all layers but last 2
image_shape = (224, 224, 3)
VGG16_MODEL = VGG16(input_shape=image_shape, #not sure this part is correct
include_top= False,
weights="imagenet"
)
VGG16_MODEL.trainable=False
pooling_layer = MaxPool2D(pool_size=(2,2),strides=2)


# 120 because there are 120 dog breeds in list, will have to change if add more breeds

last_layer = tf.keras.layers.Dense(120,activation="softmax")

model = tf.keras.Sequential([
    VGG16_MODEL,
    pooling_layer,
    Flatten(),
    last_layer
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy",metrics=['accuracy'])

# need a way to link id in training/test sets to the dog breed, might make a dictionary?

model.fit(x=training_data, validation_data=valid_data, epochs=50, verbose=2)

