import pickle as pkl
import os, shutil
import numpy as np
from multiprocessing import cpu_count

from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.optimizers import SGD

from extraction import extract_files_name, create_vgg_model, create_resnet_model,create_squeezenet_model, from_pkl_to_images, move_images, get_model_custom
from const import *

from plot import plot_roc_curve



def classWeight(labels):
	classes = np.unique(labels)
	class_weights = compute_class_weight(class_weight = 'balanced', classes = classes, y = labels) 
	class_weight = {}
	for index, weight in enumerate(class_weights):
		class_weight.update({classes[index] : weight})
	return class_weight


def train_leave_one_out(drivers, img_size, epochs, workers, batch_size, save_path, images_directory="data/Images/"):
	histories = []
	test_directory = images_directory + 'test/'
	train_directory = images_directory + 'train/'

	map_drivers_pkl = extract_files_name()
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

	i = 0
	model2 = np.array([])
	
	for driver in drivers:
		if not driver in no_test_data:
			print("Driver test : " + driver)
			i += 1

			model_name = "model_" + driver 

			save_model_path = save_path + model_name + '.{epoch:02d}-{val_loss:.2f}.hdf5'

			if 'VGG' in save_path:
				model = create_vgg_model(get_model_custom('fc6', custom_vgg_weights_path))
			elif 'ResNet' in save_path:
				model = create_resnet_model(get_model_custom('avg_pool', custom_resnet_weights_path))
			elif 'SqueezeNet' in save_path:
				model = create_squeezenet_model(get_model_custom('fire9/concat', custom_squeezeNet_weights_path))

			model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

			print("Loading the train and test images")
			if i == 1:
				map_drivers_images = from_pkl_to_images(map_drivers_pkl.values(), images_directory, driver, map_drivers_pkl)
			else:
				map_drivers_images = move_images(map_drivers_images, driver)

			print("Creation of the testing generator")
			# Test data generator
			test_datagen = ImageDataGenerator(rescale=1. / 255)
			validation_generator = test_datagen.flow_from_directory( \
						test_directory, \
						target_size = (img_size, img_size), \
						color_mode = 'rgb', \
						batch_size=batch_size, \
						class_mode='categorical', \
						shuffle=False)


			print("Creation of the training generator")
			# Train data generator
			train_datagen = ImageDataGenerator( \
				rescale=1. / 255, \
				featurewise_center=False, \
				featurewise_std_normalization=False, \
				rotation_range=10, \
				width_shift_range=0.1, \
				height_shift_range=0.1, \
				zoom_range=.1, \
				horizontal_flip=True)
			training_generator = train_datagen.flow_from_directory( \
						train_directory, \
						target_size = (img_size, img_size), \
						color_mode = 'rgb', \
						batch_size=batch_size, \
						class_mode='categorical', \
						shuffle=True)

			class_weight = classWeight(training_generator.classes)
			print("Class weight :", class_weight)

			checkpoint = ModelCheckpoint(save_model_path, monitor = 'val_loss', save_best_only = True)

			print("Training")
			hist = model.fit_generator(generator=(training_generator), validation_data=(validation_generator), epochs=epochs, steps_per_epoch = training_generator.n // batch_size+1, class_weight=class_weight, use_multiprocessing=False, workers=workers, verbose=1, callbacks = [checkpoint], validation_steps=validation_generator.n // batch_size + 1)


			histories.append(hist.history)

	return histories
