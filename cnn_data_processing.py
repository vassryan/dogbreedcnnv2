import numpy as np
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

# Image processing

ROOT="d:/csc2280/dog_breed_project"
DATAROOT = "dog_breed_project/dog-breed-identification/"


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
if os.path.isfile(os.path.join(training_path,"boston_bull","000bec180eb18c7604dcecc8fe0dba07.jpg")) is False:
    for breed in breed_list:
        for value in breed_dictionary[breed]:
            if os.path.isfile(os.path.join(training_path,value+".jpg")):
                shutil.move(os.path.join(training_path, value+".jpg"), os.path.join(training_path,breed))


# moving images into folders for test_subset
# for validation set, going to make that a subset of the training set, since labels only exist for training set
# need to be careful not to have duplicate images in validation and training, will likely have to first undo movement of files into folders temporarily
# and then take like 100-200 images from the training set and move them into validation using random.sample(glob.glob...)

# move images from breed folders back to training path so that can take sample of these images and transfer to validation path
# if os.path.isfile(os.path.join(training_path,"000bec180eb18c7604dcecc8fe0dba07.jpg")) is False:
#     for breed in breed_list:
#         for value in breed_dictionary[breed]:
#             shutil.move(os.path.join(training_path,breed, value+".jpg"), os.path.join(training_path))


# already done, don't need to run again
# if os.path.isfile(glob.glob(os.path.join(training_path,"*.jpg"))):
    # for c in random.sample(glob.glob("d:/csc2280/dog_breed_project/dog-breed-identification/train/*.jpg"),100):
    #     shutil.move(c,validation_path)

# already run so don't need to run again, need to think of if statement that will prevent from rerunning as long as no .jpg in validation folder
# for breed in breed_list:
#     for value in breed_dictionary[breed]:
#         if os.path.isfile(os.path.join(validation_path,value+".jpg")):
#             shutil.move(os.path.join(validation_path, value+".jpg"), os.path.join(validation_path,breed))






# confusion matrix





# predictions
