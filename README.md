# dogbreedcnnv2


To download the data used in this project, go to: https://www.kaggle.com/c/dog-breed-identification/data?select=test

FILE SETUP
Ideally, you want your folders to be set up such that within each dataset (training, test_subset, and validation) you have 120 folders (1 for each breed)
Each breed folder should have the appropriate dog images
The folder for the training data should be at the location of training_path, validation should be at validation_path, test should be at test_subset path

*This is likely not the optimal way of setting up the folders, but it should at least work
Steps: 
1) Change the ROOT, training_path, test_path, and validation_path to whatever you desire
2) Change the path in all of the glob.glob functions to be the appropriate path for your computer
3) Run the cnn_data_processing program 
