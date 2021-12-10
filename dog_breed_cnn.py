import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.saving import saved_model
from sklearn.metrics import confusion_matrix
import itertools
import os
import matplotlib.pyplot as plt
import warnings
import pdb
from tensorflow.python.keras.backend import relu

from tensorflow.python.keras.callbacks import EarlyStopping

warnings.simplefilter(action="ignore", category=FutureWarning)

# GPU support
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Image processing

ROOT="d:/csc2280/dog_breed_project"
DATAROOT = "dog_breed_project/dog-breed-identification/"


training_path = "d:/csc2280/dog_breed_project/dog-breed-identification/train"
test_path = "d:/csc2280/dog_breed_project/dog-breed-identification/test_subset"   #this actually goes the the test_subset folder, which contains some of the dog pictures from test
validation_path = "d:/csc2280/dog_breed_project/dog-breed-identification/validation"


# train_test_split function in keras?
# validation_split

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

# Converting images into data, separating into training, test, and validation sets
# may need to increase the number of pixels being analyzed for each image if accuracy is not high 

# Changed preprocessing function to just "preprocessing_input" rather than tf.keras.applications.vgg16.preprocess_input
training_data = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory = training_path,target_size=(224,224), classes=breed_list#make 1 folder for each breed with name of breed being the folder name
)

# pdb.set_trace()

test_data = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory = test_path,target_size=(224,224),classes=breed_list)


valid_data = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory = validation_path,target_size=(224,224),classes=breed_list
)

# alternate model that could be used in the future
# model_test = Sequential({
#     Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu", input=training_path),
#     Conv2D(filters=32, kernel_size=(3,3), padding="same",activation="relu")
#     Conv2D(filters=64, kernel_size=(3,3), padding="same",activation="relu")
#     Conv2D(filters=128, kernel_size=(3,3), padding="same",activation="relu")
# })

# Making the vgg16 model and freezing all layers but last 2
image_shape = (224, 224, 3)
VGG16_MODEL = VGG16(input_shape=image_shape, #not sure this part is correct
include_top= False,
weights="imagenet"
)

VGG16_MODEL.trainable=False

pooling_layer = MaxPool2D(pool_size=(2,2),strides=2)
wgt_reg_layer = Dense(60,kernel_regularizer=regularizers.l2(0.0001),activation=relu)
wgt_reg_layer2 = Conv2D(filters=128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(0.0001),activation="relu")
# 120 because there are 120 dog breeds in list, will have to change if add more breeds
last_layer = Dense(120,activation="softmax")

# first model I tried, overfitting of data, decreasing validation accuracy and increasing training accuracy
# model = tf.keras.Sequential([
#     VGG16_MODEL,
#     pooling_layer,
#     Flatten(),
#     last_layer
# ])

model = Sequential([
    VGG16_MODEL,
    Dense(64,activation=relu),
 #Might want to investigate BatchNormalization
    Flatten(),
    last_layer
])

model.summary()

#Changed learning rate to 0.001 from 0.0001 to hopefully reduce overfitting
model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy,metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=os.path.join(ROOT,"dog-breed-identification","vgg16_model_saves"),verbose=2,monitor="accuracy")
early_stop = EarlyStopping(monitor='val_loss', patience=3,)
call_back = [checkpoint,early_stop]

model.fit(x=training_data,batch_size=250,validation_data=valid_data, epochs=20, verbose=2,callbacks=call_back)



# keras.callbacks.ModelCheckpoint

# Might want to add weight regularization and dropout layers to reduce overfitting
# Data from first 5 epochs of first run (11/22/2021) showed 80% training acc, 34% valid acc
# Attempted again with BatchNormalization(trainable=False,epsilon=1e-9), but had really low training acc and literally 0% valid acc
# Attempt 3 involved two Conv2D layers with kernel_regularization, accuracy for training was 9% and valid was 8% after 3 epochs. Could work, but pretty low
# Attempt 4 used 1 Conv2D layer with regularization, 1% training acc, 4% valid acc after 2 epochs
# Attempt 5: 1 wgt_reg layer (Dense instead of Conv2D, used l2 weight regularization), 97.55% training acc after 11 epochs, 21% validation acc





