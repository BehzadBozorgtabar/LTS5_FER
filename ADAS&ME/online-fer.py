import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time

from extraction import create_tcnn_bottom, extract_all_face_landmarks, extract_pickle_data, get_conv_1_1_weights
from training import normalize_face_landmarks, create_tcnn_top, create_phrnn_model
from const import *

from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical

font = cv2.FONT_HERSHEY_SIMPLEX

def predict_online(frames, nb_frames_per_sample, y_true=None, merge_weight=0.5):
	plt.figure(figsize=(3,3))
	img = None

	trim = frames.shape[0] - frames.shape[0]%nb_frames_per_sample
	frames = frames[:trim]
	sequences = frames.reshape((-1, 1, nb_frames_per_sample, img_size, img_size, 3))

	y_true = np.array(y_true[:trim]).reshape((-1, nb_frames_per_sample))
	y_true = np.argmax(np.mean(to_categorical(y_true, num_classes=nb_emotions), axis=1), axis=1)

	y_preds = []

	start_time = time.time()

	# Go through all frames sequences one by one
	for i, sequence in enumerate(sequences):
		print("sequence {}/{}".format(i+1, len(sequences)), end='\r')
		im=sequence[-1, 0]
		im=np.multiply(sequence[-1, 0], 255).astype(np.uint8)#.copy() 

		# Prepare VGG-TCNN features
		vgg_tcnn_features = tcnn_bottom.predict([sequence[:,i] for i in range(nb_frames_per_sample)])

		# Prepare landmarks features for PHRNN
		landmarks = extract_all_face_landmarks(sequence[0], verbose=False)
		lm_norm = normalize_face_landmarks(landmarks)

		eyebrows_landmarks = lm_norm[:, eyebrows_mask].reshape((1, nb_frames_per_sample, -1))
		nose_landmarks = lm_norm[:, nose_mask].reshape((1, nb_frames_per_sample, -1))
		eyes_landmarks = lm_norm[:, eyes_mask].reshape((1, nb_frames_per_sample, -1))
		inside_lip_landmarks = lm_norm[:, inside_lip_mask].reshape((1, nb_frames_per_sample, -1))
		outside_lip_landmarks = lm_norm[:, outside_lip_mask].reshape((1, nb_frames_per_sample, -1))

		x_landmarks = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]

		# Compute predictions
		y_pred_tcnn = tcnn_top.predict(vgg_tcnn_features)
		y_pred_phrnn = phrnn.predict(x_landmarks)
		y_pred = merge_weight*y_pred_tcnn + (1-merge_weight)*y_pred_phrnn
		emo_pred = np.argmax(y_pred)
		y_preds.append(emo_pred)

		# Display image with predicted emotion on top
		color = (255, 0, 0)
		cv2.putText(im, emotions[emo_pred], (2, 16), font, 0.7, color, 1, cv2.LINE_AA)
		if img is None:
			img = plt.imshow(im)
		else:
			img.set_data(im)

		plt.pause(.1)
		plt.draw()

	elapsed_time = time.time() - start_time
	avg_time_per_predictions = elapsed_time/len(sequences)

	print('')
	print('Finished online-predictions')
	print(' Average time per prediction : {:3f} seconds'.format(avg_time_per_predictions))

	# Compute the accuracy of the emotion prediction over the sequence
	if y_true is not None:
		acc = (y_true == y_preds).sum()/len(y_preds)
		print(' Accuracy over image sequence : {:.3f}'.format(acc))


#### MAIN CODE ####

sift_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/sift-phrnn1_Driver2.h5'
lm_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ADAS&ME/models/late_fusion/phrnn/lm-phrnn1_Driver2.h5'
tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/tcnn1_Driver2.h5'

# Prepare TCNN model
conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)
tcnn_top = create_tcnn_top()()
tcnn_top.load_weights(tcnn_model_path)

# Prepare PHRNN model
phrnn = create_phrnn_model(features_per_lm=2)()
phrnn.load_weights(lm_phrnn_model_path)

pickle_file_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/Real_data/TS7_DRIVE/20180829_114712_1.pkl'

# Get data and lauch online prediction
nb_frames = 100
frames, y_true, _ = extract_pickle_data(pickle_file_path)
frames = frames[:nb_frames]
y_true = y_true[:nb_frames]

predict_online(frames, nb_frames_per_sample=5, y_true=y_true)

