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
from training import create_tcnn_bottom, get_conv_1_1_weights, create_phrnn_model, create_tcnn_top, train_tcnn_crossval, train_phrnn_crossval, load_cross_val_models, cross_validate_tcnn_phrnn, CustomVerbose
from sift import extract_all_sift_features
from plot import plot_histories, plot_confusion_matrix
from evaluate import evaluate_tcnn_phrnn_model
from const import *

###########################
######## MAIN CODE ########
###########################

# Train the model only if the call was made with argument 'train',
# otherwise, just test it
train = len(sys.argv) > 1 and sys.argv[1]=='train'

# If the sift or vgg features were pre-computed, we don't have to recompute them
load_vgg_features = os.path.isfile(vgg_features_tcnn_path)
load_sift_features = os.path.isfile(sift_features_path)
load_frames = os.path.isdir(frames_path)


## EXTRACTION ##

# If not already done, we extract the relevant frames from the raw MUG dataset
if not load_frames:
	extract_frames(subjects_path, frames_path)


# Now we extract the training and target data from the frames
if not(load_vgg_features and load_sift_features):
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
	conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
	tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)

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

if load_sift_features:
	with open(sift_features_path, 'rb') as f:
		sift_features = pickle.load(f)
else:
	# Extract SIFT features
	sift_features = extract_all_sift_features(x)

sift_features = sift_features.reshape((-1, 5, 51, 128))

# Prepare the different landmarks clusters
eyebrows_landmarks = sift_features[:,:, eyebrows_mask].reshape((nb_samples, nb_frames, -1))
nose_landmarks = sift_features[:,:, nose_mask].reshape((nb_samples, nb_frames, -1))
eyes_landmarks = sift_features[:,:, eyes_mask].reshape((nb_samples, nb_frames, -1))
inside_lip_landmarks = sift_features[:,:, inside_lip_mask].reshape((nb_samples, nb_frames, -1))
outside_lip_landmarks = sift_features[:,:, outside_lip_mask].reshape((nb_samples, nb_frames, -1))

sift_inputs = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]

## TRAINING ##

if train:
	print("Training TCNN model...")
	batch_size=32
	epochs=80
	n_splits=5
	save_best_model = True
	trained_tcnn_model_path = 'models/tcnn/tcnn2.h5'

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
												   model_path=trained_tcnn_model_path)
	print("\nTraining of TCNN complete.")
	plot_histories(histories, 'VGG-TCNN, {}-fold cross-validation'.format(n_splits))


	print("Training PHRNN model...")
	batch_size=32
	epochs=80
	n_splits=5
	save_best_model = True
	trained_phrnn_model_path = 'models/phrnn/sift-phrnn2.h5'

	# Create the callbacks
	custom_verbose = CustomVerbose(epochs)
	early_stop = EarlyStopping(patience=30)
	callbacks = [custom_verbose, early_stop]

	phrnn, skf, histories = train_phrnn_crossval(create_phrnn_model(features_per_lm=128), 
	                                             sift_inputs, 
	                                             y, 
	                                             batch_size=batch_size, 
	                                             epochs=epochs, 
	                                             callbacks=callbacks, 
	                                             n_splits=n_splits, 
	                                             save_best_model=save_best_model, 
	                                             model_path=trained_phrnn_model_path)
	print("\nTraining of PHRNN complete.")
	plot_histories(histories, 'PHRNN, {}-fold cross-validation'.format(n_splits))



# TESTING ##
tcnn_model_path = trained_tcnn_model_path if train else tcnn_model_path
phrnn_model_path = trained_phrnn_model_path if train else sift_phrnn_model_path

y_pred, y_true = evaluate_tcnn_phrnn_model(vgg_features, sift_inputs, y, tcnn_model_path, phrnn_model_path, n_splits=5, merge_weight=0.4)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, emotions, title='TCNN-SIFT-PHRNN  -  MUG', normalize=True)
