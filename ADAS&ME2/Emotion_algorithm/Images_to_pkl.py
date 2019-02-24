import os
import numpy as np
import pickle
import cv2

from const import *
from extraction import create_directory

emotions_folders = {'Neutral' : 1, 'Positive' : 2, 'Frustrated' : 3, 'Anxiety' : 4}

def save_pkl(pkl_path, annots, images):
	annots = np.array(annots)
	images = np.array(images)
	with open(pkl_path + ".pkl", "wb") as file:
		pickle.dump((images, annots), file)

for folder in os.listdir(images_files_path):
	path = images_files_path + folder + '/'
	pkl_directory = real_data_path + folder + '/'
	pkl_path = pkl_directory + folder
	
	create_directory(pkl_directory)
	
	images = []
	annots = []
	counter = 0
	for emotion in os.listdir(path):
		emotion_path = path + emotion + '/'
		for image in os.listdir(emotion_path):
			print("Extracting image %d" % counter, end = '\r')
			counter += 1

			image_path = emotion_path + image
			severity = emotions_folders[emotion]
			cameraIndex = 0
			frameNumber = counter

			images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
			annots.append(({'CameraIndex': cameraIndex, 'FrameNumber' : frameNumber, 'Severity' : severity}))

			if counter % 1000 == 0:
				save_pkl(pkl_path + str(counter // 1000), annots, images)
				annots = []
				images = []

	save_pkl(pkl_path + str((counter // 1000) + 1), annots, images)
