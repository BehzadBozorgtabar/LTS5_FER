import sys
import os, shutil
import pickle as pkl
import multiprocessing
import numpy as np

import matplotlib.pyplot as plt

from train import train_leave_one_out
from test import evaluate_one_driver_model, plot_full_evaluation
from const import *

from imageio import imread, imwrite

from extract_smb_frames import extract_smb_frames
from extraction import create_directory, extract_test_images

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

nbr_args = len(sys.argv)

if nbr_args > 2:
	action = sys.argv[1]
	model_name = sys.argv[2]
	good_answers = ['train', 'test']
	if not action in good_answers or not model_name in ['VGG', 'ResNet', 'SqueezeNet']:
		print("Error, wrong arguments")
		sys.exit()
	
else:
	print("Error, you must specify if you want to train the model or test it and which model you want to use.")
	sys.exit()

train = action == 'train'

if model_name == 'VGG':
	save_path = vgg_model_path
	img_size = 224
elif model_name == 'ResNet':
	save_path = resnet_model_path
	img_size = 224
elif model_name == 'SqueezeNet':
	save_path = squeezeNet_model_path
	img_size = 227

epochs = 50
workers = 4

if train:
	drivers_test = os.listdir(real_data_path)
	to_fine_tune = None

	if nbr_args == 5:
		to_fine_tune = sys.argv[3]
		drivers_test = sys.argv[4]

	histories = train_leave_one_out(drivers_test, img_size, epochs, workers, batch_size=32, save_path=save_path, to_fine_tune=to_fine_tune)
	with open(results_path + "histories.pkl", 'wb') as pklfile:
		pkl.dump(histories, pklfile)
else:
	models = [x for x in os.listdir(save_path) if x.endswith('hdf5')]

	all_y_test = []
	all_y_pred = []

	for model in models:
		
		y_test, y_pred = evaluate_one_driver_model(model, model_name, save_path, img_size=img_size, batch_size=32)

		all_y_test.append(y_test)
		all_y_pred.append(y_pred)

	all_y_test = np.concatenate(all_y_test, axis = 0)
	all_y_pred = np.concatenate(all_y_pred, axis = 0)

	cm, comments = plot_full_evaluation(all_y_test, all_y_pred, model_name)
	print(cm)
	print(comments)
