import os, shutil
import numpy as np
import csv
import dlib
import cv2
import pickle as pkl

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout, Activation, Convolution2D, GlobalAveragePooling2D

from imageio import imread, imwrite

from const import *

#### Read a CSV file ####
def readCSV(csvPath):
	data = []
	with open(csvPath, 'r') as csvFile:
		spamReader = csv.DictReader(csvFile)
		for row in spamReader:
			data.append(row)
	return np.array(data)

#### Create a new directory ####
def create_directory(path):
	if not os.path.exists(path):
		os.makedirs(path)

### Extract test images only ###
def extract_test_images(test_files, driver_test_path, test_directory):
	for i in range(nb_emotions):
		directory_to_create = test_directory + str(i+1) + "/"
		if os.path.exists(directory_to_create):
			shutil.rmtree(directory_to_create)
		create_directory(directory_to_create)

	for pkl_file in test_files:
		with open(driver_test_path + pkl_file, 'rb') as file:
			images, annotations = pkl.load(file)
			for i, image in enumerate(images):
				annot = annotations[i]
				severity = annot['Severity']
				frame_number = annot['FrameNumber']
				dest_path = test_directory + str(severity) + '/' + str(frame_number) + '.png'
				imwrite(dest_path, image)

### Move images from test to train or train to test ###
def move_images(map_drivers_images, driver_test):
	new_map = {}
	for driver in map_drivers_images:
		new_map.update({driver : []})
		test = driver == driver_test
		move_to = 'test' if  test else 'train'
		for image in map_drivers_images[driver]:
			split_path = image.split('/')
			prev_directory = split_path[2]
			new_image = image
			if (test and prev_directory == 'train') or (not test and prev_directory == 'test'):
				split_path[2] = move_to
				new_image = '/'.join(split_path)
				os.rename(image, new_image)
			
			new_map[driver].append(new_image)

	return new_map
			

### Extract images from pkl files and sort them by the severity annotations ### 
def from_pkl_to_images(pkl_files, images_directory, driver_test, map_drivers_pkl):
	
	map_drivers_images = {}
	
	for i in range(nb_emotions):
		directory_to_create = images_directory + 'test/' + str(i+1) + "/"
		if os.path.exists(directory_to_create):
			shutil.rmtree(directory_to_create)
		create_directory(directory_to_create)

		directory_to_create = images_directory + 'train/' + str(i+1) + "/"
		if os.path.exists(directory_to_create):
			shutil.rmtree(directory_to_create)
		create_directory(directory_to_create)

	j = 0
	for driver in map_drivers_pkl:
		save_path = images_directory + 'test/' if driver == driver_test else images_directory + 'train/'
		map_drivers_images.update({driver : []})
		for pkl_file in map_drivers_pkl[driver]:
			with open(pkl_file, 'rb') as file:
				images, annotations = pkl.load(file)
				for i, image in enumerate(images):
					annot = annotations[i]
					severity = annot['Severity']
					frame_number = annot['FrameNumber']
					directory = save_path + str(severity) + '/'
					dest_path = directory + str(j*1000000000 + frame_number) + '.png'

					map_drivers_images[driver].append(dest_path)
					imwrite(dest_path, image)
			j += 1

	return map_drivers_images


### Real data files extraction ###
def extract_files_name():
	dico = {}

	path = real_data_path
	for folder in os.listdir(path):
		dico.update({folder : []})
		folder_path = path + folder + '/'
		for pkl in os.listdir(folder_path):
			pkl_path = folder_path + pkl
			if pkl.endswith(".pkl") and "vgg" not in pkl and "sift" not in pkl:
				dico[folder].append(pkl_path)
	return dico

#### SQUEEZENET FINE_TUNED MODEL ####
def create_squeezenet_model(model):

	#for layer in model.layers:
	#		layer.trainable = False
	x = Dropout(0.5, name='drop9')(model.layers[-1].output)
	x = Convolution2D(nb_emotions, (1, 1), padding='valid', name='convLast')(x)
	x = Activation('relu', name='relu_convLast')(x)
	x = GlobalAveragePooling2D()(x)
	output = Activation('softmax', name='loss')(x)
	
	custom_model = Model(inputs = model.input, outputs = output)
	return custom_model

#### RESNET FINE-TUNED MODEL ####
def create_resnet_model(model):

	#for layer in model.layers:
	#		layer.trainable = False
	x = Flatten(name = 'flatten')(model.layers[-1].output)
	x = Dense(512, activation = 'relu', name = 'fc7')(x)
	output = Dense(nb_emotions, activation = 'softmax', name = 'output')(x)

	custom_model = Model(inputs = model.input, outputs = output)

	return custom_model

#### VGG FINE-TUNED MODEL ####
def create_vgg_model(model):

	#for layer in model.layers:
	#		layer.trainable = False

	x = Dense(512, activation = 'relu', name = 'fc7')(model.layers[-1].output)
	output = Dense(nb_emotions, activation = 'softmax', name = 'output')(x)

	custom_model = Model(inputs = model.input, outputs = output)

	return custom_model

#### VGG MODEL ####
def get_model_custom(output_layer, pre_trained_model):
	custom_model = load_model(pre_trained_model)
	if output_layer is not None:
		output = custom_model.get_layer(output_layer).output
		custom_model = Model(inputs=custom_model.input, outputs=output)

	return custom_model
