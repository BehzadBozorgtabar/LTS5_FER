import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, Flatten, Dropout, Activation
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers

from extract import extract_frames, extract_data, extract_model_features
from training import create_tcnn_bottom, get_conv_1_1_weights, create_phrnn_model, train_tcnn_crossval, train_phrnn_crossval, load_cross_val_models, cross_validate_tcnn_phrnn, CustomVerbose
from sift import compute_face_landmarks, normalize_face_landmarks
from plot import plot_histories, plot_confusion_matrix
from const import *

def create_tcnn_top():
	"""Create the top of the tcnn with fully connected layers.
	"""
	input_shape=(7, 7, 512)

	tcnn_top = Sequential()
	tcnn_top.add(Convolution2D(1024, (7, 7), activation='relu', name='fc6', input_shape=input_shape))
	tcnn_top.add(Dropout(0.5))
	tcnn_top.add(Convolution2D(512, (1, 1), activation='relu', name='fc7'))
	tcnn_top.add(Dropout(0.5))
	tcnn_top.add(Convolution2D(7, (1, 1), name='fc8'))
	tcnn_top.add(Flatten())
	tcnn_top.add(Activation('softmax'))
	
	tcnn_top.compile(loss='categorical_crossentropy',
				 optimizer=optimizers.Adam(),
				 metrics=['accuracy'])
	
	return tcnn_top

###########################
######## MAIN CODE ########
###########################

# Train the model only if the call was made with argument 'train',
# otherwise, just test it
train = len(sys.argv) > 1 and sys.argv[1]=='train'

# If the sift or vgg features were pre-computed, we don't have to recompute them
load_vgg_features = os.path.isfile(vgg_features_tcnn_path)
load_face_landmarks = os.path.isfile(face_landmarks_path)
load_frames = os.path.isdir(frames_path)


## EXTRACTION ##

# If not already done, we extract the relevant frames from the raw MUG dataset
if not load_frames:
	extract_frames(subjects_path, frames_path)


# Now we extract the training and target data from the frames
if not(face_landmarks_path and load_vgg_features):
	x, y = extract_data(frames_path)
else:
	with open(y_data_path, 'rb') as f:
		y = pickle.load(f)
nb_samples = len(y)


if load_vgg_features:
	with open(vgg_features_tcnn_path, 'rb') as f:
		vgg_features = pickle.load(f)
else:
	print('Extracting VGG features...')
	# Create VGG model
	conv1_1_weigths = get_conv_1_1_weights(vgg_weights_path)
	tcnn_bottom = create_tcnn_bottom(vgg_weights_path, conv1_1_weigths)

	# Extract the VGG features
	vgg_features = tcnn_bottom.predict([x[:,i] for i in range(nb_frames)])


# Define the masks corresponding to each cluster of landmarks
lm_range = np.array(range(51))
eyebrows_mask = (lm_range < 10) # 10 landmarks
nose_mask = np.logical_and(10 <= lm_range, lm_range < 19) # 9 landmarks
eyes_mask = np.logical_and(19 <= lm_range, lm_range < 31)# 12 landmarks
outside_lip_mask = np.logical_and(31 <= lm_range, lm_range < 43) # 12 landmarks
inside_lip_mask = (43 <= lm_range) # 8 landmarks
nose_landmark_idx = 13

if load_face_landmarks:
	with open(face_landmarks_path, 'rb') as f:
		face_landmarks = pickle.load(f)
else:
	print('Computing face landmarks...')
	face_landmarks = compute_face_landmarks(x)
face_landmarks_norm = normalize_face_landmarks(face_landmarks, nose_landmark_idx)

# Prepare the different landmarks clusters
eyebrows_landmarks = face_landmarks_norm[:,:, eyebrows_mask].reshape((nb_samples, nb_frames, -1))
nose_landmarks = face_landmarks_norm[:,:, nose_mask].reshape((nb_samples, nb_frames, -1))
eyes_landmarks = face_landmarks_norm[:,:, eyes_mask].reshape((nb_samples, nb_frames, -1))
inside_lip_landmarks = face_landmarks_norm[:,:, inside_lip_mask].reshape((nb_samples, nb_frames, -1))
outside_lip_landmarks = face_landmarks_norm[:,:, outside_lip_mask].reshape((nb_samples, nb_frames, -1))

landmarks_inputs = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]

## TRAINING ##

if train:
	print("Training TCNN model...")
	batch_size=32
	epochs=80
	n_splits=5
	save_best_model = False
	model_path = 'models/tcnn/tcnn2.h5'

	# Create the callbacks
	custom_verbose = CustomVerbose(epochs)
	early_stop = EarlyStopping(patience=30)
	callbacks = [custom_verbose, early_stop]

	tcnn_top, skf, histories = train_tcnn_crossval(create_tcnn_top, 
												   vgg_features, 
												   y, 
												   batch_size=batch_size, 
												   epochs=epochs, 
												   callbacks=callbacks, 
												   n_splits=n_splits, 
												   save_best_model=save_best_model, 
												   model_path=model_path)
	print("\nTraining of TCNN complete.")
	plot_histories(histories, 'VGG-TCNN, {}-fold cross-validation'.format(n_splits))


	print("Training PHRNN model...")
	batch_size=32
	epochs=80
	n_splits=5
	save_best_model = False
	model_path = 'models/phrnn/phrnn2.h5'

	# Create the callbacks
	custom_verbose = CustomVerbose(epochs)
	early_stop = EarlyStopping(patience=30)
	callbacks = [custom_verbose, early_stop]

	phrnn, skf, histories = train_phrnn_crossval(create_phrnn_model, 
	                                             landmarks_inputs, 
	                                             y, 
	                                             batch_size=batch_size, 
	                                             epochs=epochs, 
	                                             callbacks=callbacks, 
	                                             n_splits=n_splits, 
	                                             save_best_model=save_best_model, 
	                                             model_path=model_path)
	print("\nTraining of PHRNN complete.")
	plot_histories(histories, 'PHRNN, {}-fold cross-validation'.format(n_splits))


# Load all models to cross-validation
tcnn_model_path = 'models/tcnn/tcnn'+('2' if train else '')
phrnn_model_path = 'models/phrnn/phrnn'+('2' if train else '')
model_path_ext = '.h5'
tcnn_models, phrnn_models = load_cross_val_models(tcnn_model_path, phrnn_model_path, model_path_ext, 5, y)

merge_weights = [0.5]
accuracies = cross_validate_tcnn_phrnn(merge_weights, 5, tcnn_models, phrnn_models, vgg_features, landmarks_inputs, y)
best_crossval_acc = np.average(accuracies, axis=0)[0]
print('Cross-validation accuracy of TCNN-PHRNN : {:.4f}'.format(best_crossval_acc))
