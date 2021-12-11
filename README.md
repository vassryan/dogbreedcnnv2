# dogbreedcnnv2

To download the data used in this project, go to: https://www.kaggle.com/c/dog-breed-identification/data?select=test

Before you begin, you need to make sure the folders are set up in the following way:

Within the dog_breed_project folder, you should have the dog-breed-identification folder and each of the python files necessary for the project (cnn_data_processing, dog_breed_cnn, model_prediction, and README)

Within the dog-breed-identification folder, you should have folders named "test", "test_subset", "train", "validation". You should also have dog_breed_labels_ids in this folder. 

In the "train" folder, you should place all of the training images from the kaggle dataset. All of the test images from the kaggle dataset should be placed within the "test" folder. 

FILE SETUP
Ideally, you want your folders to be set up such that within each dataset (training and validation) you have 120 folders (1 for each breed)
Each breed folder should have the appropriate dog images
The folder for the training data should be at the location of training_path, validation should be at validation_path, test should be at test_subset path

*This is likely not the optimal way of setting up the folders, but it should at least work

Steps: 
1) Change the ROOT, training_path, test_path, and validation_path to whatever you desire
2) Change the path in all of the glob.glob functions to be the appropriate path for your computer
3) Run the cnn_data_processing program 
4) Enjoy!


Now that the data is set up, you can choose to run the dog_breed_cnn.py file if you desire. This will initiate training of the model on your computer and will save the model after each epoch. Alternatively, you can use the already trained model called saved_model.pb which I have added to the github repository.
If you choose to use the saved_model, then make sure you place it within a folder called "vgg16_model_saves2" in the dog-breed-identification folder (you will have to create this folder yourself)

To predict using this model, run the model_prediction program. 