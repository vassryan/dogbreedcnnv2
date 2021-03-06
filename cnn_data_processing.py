import numpy as np
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import pdb

warnings.simplefilter(action="ignore", category=FutureWarning)

# Image processing

ROOT="d:/csc2280/dog_breed_project"
DATAROOT = "dog_breed_project/dog-breed-identification/"


training_path = "d:/csc2280/dog_breed_project/dog-breed-identification/train"
test_path = "d:/csc2280/dog_breed_project/dog-breed-identification/test_subset"   #this actually goes the the test_subset folder, which contains some of the dog pictures from test
validation_path = "d:/csc2280/dog_breed_project/dog-breed-identification/validation"

# Moves 1000 images from the full test set to the test_subset folder
def create_test_subset():
    for c in random.sample(glob.glob("d:/csc2280/dog_breed_project/dog-breed-identification/test/*"),1000):
        shutil.move(c,test_path)

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

if os.path.isdir(os.path.join(validation_path,"affenpinscher")) is False:
    for i in range(len(breed_dictionary.keys())):
            os.makedirs(os.path.join(validation_path, breed_list[i]))

def move_training_to_validation():
    if os.path.isfile(glob.glob(os.path.join(training_path,"*.jpg"))):
        for c in random.sample(glob.glob("d:/csc2280/dog_breed_project/dog-breed-identification/train/*.jpg"),1000):
            shutil.move(c,validation_path)

# add correct dog breeds to appropriate breed folders in the training folder
def move_imgs_to_training_breed_folders():
    for breed in breed_list:
        for value in breed_dictionary[breed]:
            if os.path.isfile(os.path.join(training_path,value+".jpg")):
                shutil.move(os.path.join(training_path, value+".jpg"), os.path.join(training_path,breed))


# moving images into folders for test_subset
# for validation set, going to make that a subset of the training set, since labels only exist for training set
# need to be careful not to have duplicate images in validation and training, will likely have to first undo movement of files into folders temporarily
# and then take like 100-200 images from the training set and move them into validation using random.sample(glob.glob...)

def move_imgs_validation_to_validation_breed_folder(): #moves images in the validation folder to the validation breed folders
    for breed in breed_list:
        for value in breed_dictionary[breed]:
            if os.path.isfile(os.path.join(validation_path,value+".jpg")):
                shutil.move(os.path.join(validation_path, value+".jpg"), os.path.join(validation_path,breed))


move_training_to_validation()

move_imgs_to_training_breed_folders()

move_imgs_validation_to_validation_breed_folder()

create_test_subset()