import sys
import os, shutil
import pickle as pkl
import multiprocessing
import numpy as np

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from extraction import extract_test_images
from plot import plot_confusion_matrix
from const import *

### Evaluate the model for all driver ###
def plot_full_evaluation(all_y_test, all_y_pred, model_name):
	final_result = confusion_matrix(all_y_test , all_y_pred, labels= np.arange(nb_emotions))

	accuracy = (all_y_test == all_y_pred).sum() / len(all_y_pred)
	average = 'macro'

	precision = precision_score(all_y_test, all_y_pred, average=average)
	recall = recall_score(all_y_test, all_y_pred, average=average)
	f1 = f1_score(all_y_test, all_y_pred, average=average)

	comments = ['\nMODEL EVALUATION:']
	comments.append('  Accuracy  : {:.5f}'.format(accuracy))
	comments.append('  Precision : {:.5f}'.format(precision))
	comments.append('  Recall    : {:.5f}'.format(recall))
	comments.append('  F1-score  : {:.5f}'.format(f1))

	comments = '\n'.join(comments)

	plot_confusion_matrix(final_result, emotions, title="Confusion_Matrix_" + model_name, normalize=True, comments=comments)

	return final_result, comments

### Evaluate the model of a specific driver ###
def evaluate_one_driver_model(model, model_name, save_path, img_size=224, batch_size=5, workers=4):
	driver_test = model[6:-13]
	print("Test driver",driver_test)
	
	model = load_model(save_path + model)
	driver_test_path = real_data_path + driver_test + '/'

	test_files = [x for x in os.listdir(driver_test_path)]
	images_directory = "data/Images/"
	test_directory = images_directory + 'test/'
	
	print("Extraction of the images")
	extract_test_images(test_files, driver_test_path, test_directory)	

	print("Creation of the test generator")
	test_datagen = ImageDataGenerator(rescale = 1./255)
	test_generator = test_datagen.flow_from_directory( \
					test_directory, \
					target_size = (img_size, img_size), \
					color_mode = 'rgb', \
					batch_size=batch_size, \
					class_mode='categorical', \
					shuffle=False)

	print("Predictions")
	predictions = model.predict_generator(generator=test_generator, use_multiprocessing=False, workers=workers, verbose=1, steps=test_generator.n // batch_size + 1)

	y_pred = np.argmax(predictions, axis = 1)
	y_test = test_generator.classes
	results = confusion_matrix(y_test , y_pred, labels= np.arange(nb_emotions))
	plot_confusion_matrix(results, emotions, title="Confusion_Matrix_" + driver_test + "_" + model_name, normalize=True)

	return y_test, y_pred
