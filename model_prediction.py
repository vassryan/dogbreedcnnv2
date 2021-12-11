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
import pdb

ROOT="d:/csc2280/dog_breed_project"
test_path = "d:/csc2280/dog_breed_project/dog-breed-identification/test_subset"


new_model = load_model("d:/csc2280/dog_breed_project/dog-breed-identification/vgg16_model_saves2/")

new_model.summary()


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
    directory = test_path,target_size=(224,224), shuffle=False, classes=["all_class"])

filenames = test_data.filenames
nb_samp = len(filenames)
# pdb.set_trace()

test_samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


samples_predict = []

predictions = new_model.predict(x=test_data, batch_size=10, verbose=0)

class_predictions = np.argmax(predictions,axis=1)

for i in range(len(test_data)):
    print("X = {}, Predicted = {}".format(breed_list[test_data[i]],breed_list[class_predictions[i]]))

# predictions = new_model.predict(x=test_data,verbose=0)
# new_model.predict
# np.round(predictions)

# conf_mat = confusion_matrix(y_true=test_data.classes, y_pred=np.argmax(predictions,axis=-1))
# print(conf_mat)

# def plot_confusion_matrix(cm, classes,normalize=False,title="Confusion Matrix", cmap=plt.cm.Blues):
#     pass


