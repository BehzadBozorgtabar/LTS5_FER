#TRAIN and TEST guide from facial emotion recognition CNNs

First, you must remove the .gitignore file under each folder.

#TRAIN model

#A) Load the dataset

	#Annotated CSV files

1) Put the annotated csv files and the corresponding smb files in separate folders for each subject if you want to test them separatly. Otherwise, you must put the files in a same folder.

2) Please, put the folders in the "Annotations_to_pkl/" folder.

3) Run the script "Annotations_to_pkl.py", It will create pkl files for each Subject S in the "data/Real_data/S/" folder. The data are ready to be trained and tested. 


	#Annotated Images

1) Put the different folders (Neutral, Positve, Frustrated, Anxiety) containing the images in seperate folders for each subject if you want to test them separatly. Otherwise, put them in a same folder. Be careful, the name of the image must be a frame number.

2) Please, put the folders in the "Images_to_pkl folder/".

3) Run the script "Images_to_pkl.py", It will create pkl files for each Subject in the "data/Real_data/Subject" folder. The data are ready to be trained and tested.


#B) Run the script to train

	#Train all, leave one out: you want to fine-tune the model pretrained with RAF dataset.

1) Decide, which subject you don't want to test and write them in the list "no_test_data" in the const.py file

2) Run: python3 main.py train ALGORITHM_TYPE
	With ALGORITHM_TYPE either equal to VGG or SqueezeNet or ResNet

	#Fine-tune a trained model

1) Run: python3 main.py train ALGORITHM_TYPE model_path test_driver
	With model_path the path of the model, test_driver the driver tested in the model


#TEST model

1) Keep the images used to train in the "data/Real_data/" folder

2) Put the models you want to test in the "ALGORITHM_TYPE_model/" folder with ALGORITHM_TYPE either equal to VGG or SqueezeNet or ResNet

3) Run: python3 main.py test AlGORITHM_TYPE

4) Retrieve the results in the "results/" folder
