import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave
from scipy.misc import imresize
from keras.utils.np_utils import to_categorical

from const import *


def extract_frames(subjects_path, frames_path, plot_statistics=False):
	"""Extracts and pre-process the relevent frames from the raw MUG dataset and write them on disk.
	
	Keyword arguments:
	subjects_path -- path of the folder containing the subjects of the MUG dataset
	frames_path -- path of the folder in which the selected frames should be stored
	plot_statistics -- if True, plot some statistics about the MUG dataset
	"""
	subjects = [s for s in os.listdir(subjects_path) if s[0]!='.']
	nb_subjects = len(subjects)

	# Initialize variables used get some insight into the dataset
	nb_frames_total = 0
	nb_takes = 0
	emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
	nb_takes_by_emo = {'anger':0, 'disgust':0, 'fear':0, 'happiness':0, 'neutral':0, 'sadness':0, 'surprise':0}
	nb_takes_by_subj = {}

	padding_ratio = 0.25

	for (i, subj) in enumerate(subjects):
		nb_takes_by_subj[subj] = 0
		subj_path = subjects_path+'/'+subj

		emotions = [e for e in os.listdir(subj_path) if e in nb_takes_by_emo.keys()]
		for emotion in emotions:
			emotion_path = subj_path+'/'+emotion

			takes = [t for t in os.listdir(emotion_path) if t[0]!='.' and not t.endswith('end')]
			nb_takes += len(takes)
			nb_takes_by_emo[emotion] += len(takes)
			nb_takes_by_subj[subj] += len(takes)

			for take in takes:
				take_path = emotion_path+'/'+take
				frames = sorted([i for i in os.listdir(take_path) if i.endswith('.jpg')])
				nb_frames_total += len(frames)
				first_frame_idx = int(frames[0][4:8])

				# Select a given number of equally spaced frames, with the given padding
				linspace = np.linspace(int(padding_ratio*len(frames)), int((1-padding_ratio)*(len(frames)-1)), nb_frames)
				select_frames_idx = [first_frame_idx+int(round(i)) for i in linspace]
				select_frames_name = ['img_{0:04d}.jpg'.format(i) for i in select_frames_idx]

				# Copy the selected frames in the given folder
				for n in select_frames_name:
					# Load the image and resize it
					src = subjects_path+'/'+subj+'/'+emotion+'/'+take+'/'+n
					img = imread(src)
					img = imresize(img, (img_size, img_size))

					# Save resize image in destination folder
					dst_folder = frames_path+'/'+emotion+'/'+subj+'_'+take
					dst = dst_folder+'/'+n
					if not os.path.exists(dst_folder):
						os.makedirs(dst_folder)
						
					imsave(dst, img)

		print("Extracting frames... {:.1f}%".format(100.*(i+1)/nb_subjects), end='\r')            

	if plot_statistics:
		print('\n{} subjects'.format(nb_subjects))
		print('{} takes'.format(nb_takes))
		print('{} frames'.format(nb_frames_total))

		plt.bar(range(len(nb_takes_by_emo)), list(nb_takes_by_emo.values()), align='center')
		plt.xticks(range(len(nb_takes_by_emo)), list(nb_takes_by_emo.keys()))
		plt.ylabel('Number of takes')
		plt.xlabel('Emotions')
		plt.show()

		plt.bar(range(len(nb_takes_by_subj)), list(nb_takes_by_subj.values()), align='center')
		plt.ylabel('Number of takes')
		plt.xlabel('Subjects')
		plt.show()


def extract_data(frames_path):
	"""Extracts training and target data (x & y) from the video frames.
	
	Keyword arguments:
	frames_path -- path of the folder in which the selected frames should be stored
	"""
	x = []
	y = []

	for emo in emotions:
		takes = [t for t in os.listdir(frames_path+"/"+emo) if t[0]!='.']
		for take in takes:
			sequence = []
			frames = sorted([i for i in os.listdir(frames_path+"/"+emo+"/"+take) if i.endswith('.jpg')])
			for frame in frames:
				img = imread(frames_path+"/"+emo+"/"+take+"/"+frame)
				sequence.append(img)
			x.append(np.asarray(sequence))
			y.append(emotions.index(emo))
	
	x = np.multiply(np.array(x), 1/256)
	y = to_categorical(np.asarray(y))
	
	return x, y


def split_train_test(x, perc_train):
	"""Splits the training and target data so that each emotion is prepertionnaly equally represented.
	"""
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	
	#for i in range(len(x)):
	for i in range(len(x)):
		nb_samples = x[i].shape[0]
		cut = int(perc_train*nb_samples)
		x_train.append(x[i][:cut])
		x_test.append(x[i][cut:])
		y_train.append(np.full((cut,), i))
		y_test.append(np.full((nb_samples-cut,), i))
		
	x_train = np.concatenate(x_train)
	x_test = np.concatenate(x_test)
	y_train = to_categorical(np.concatenate(y_train))
	y_test = to_categorical(np.concatenate(y_test))
	
	return x_train, x_test, y_train, y_test

def extract_model_features(model, img_sequences):
	"""Computes the the output features of the given model for the given image sequences.
	"""
	features = np.array([model.predict(img_sequences[:, i]) for i in range(img_sequences.shape[1])])
	features = np.swapaxes(features, 0, 1)
	return features


