#Predictions made by model
import os
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
new_model = load_model("D:/csc2280/dog_breed_project/dog-breed-identification/vgg16_model_saves2/saved_model.pb")

ROOT="d:/csc2280/dog_breed_project"
test_path = "d:/csc2280/dog_breed_project/dog-breed-identification/test_subset" 

breed_dictionary ={}
label_dict = open(os.path.join(ROOT,"dog-breed-identification","dog_breed_labels_ids.txt"),"r")
for line in label_dict:
    line = line.split()
    breed_id = line[0]
    breed_name = line[1]
    breed_dictionary.setdefault(breed_name,[]).append(breed_id)
label_dict.close()

breed_list= list(breed_dictionary.keys())

test_data = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    directory = test_path,target_size=(224,224),classes=breed_list)

predictions = new_model.predict(x=test_data,verbose=0)

np.round(predictions)

conf_mat = confusion_matrix(y_true=test_data.classes, y_pred=np.argmax(predictions,axis=-1))
print(conf_mat)
def plot_confusion_matrix(cm, classes,normalize=False,title="Confusion Matrix", cmap=plt.cm.Blues):
    pass


