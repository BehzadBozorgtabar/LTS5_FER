import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time
from imageio import imread, imwrite
from scipy.misc import imresize

from extraction import create_tcnn_bottom, extract_all_face_landmarks, extract_pickle_data, get_conv_1_1_weights, read_image, read_roi_data, read_smb_header, get_face_square
from training import normalize_face_landmarks, create_tcnn_top, create_phrnn_model
from const import *

from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical

font = cv2.FONT_HERSHEY_SIMPLEX

def get_frame_from_smb(smb_file_path, frame_idx):
	file = open(smb_file_path, "rb")
	SMB_HEADER_SIZE = 20
	subj_frames = []
	i = 0
	image=None
	try:
		# Read SMB header
		image_width, image_height = read_smb_header(file)

		current_position = frame_idx * ((image_width * image_height) + SMB_HEADER_SIZE)
		file.seek(current_position)

		smb_header = file.read(SMB_HEADER_SIZE)
		if not smb_header:
			raise Exception('End of the smb file') 

		# Read ROI data
		camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(file)

		# Read image
		file.seek(current_position + SMB_HEADER_SIZE)
		image = read_image(file, image_width, image_height)

		# Align image with face using roi data
		((l,t),(r,b)) = get_face_square(roi_left, roi_top, roi_width, roi_height, 1)
		image = image[t:b,l:r]

		# Resize image to 224x224
		image = imresize(image, (img_size, img_size))

		# convert to rgb if greyscale
		if len(image.shape)==2:
			image = np.stack((image,)*3, axis=-1)
	finally:
		file.close()

	return image

def get_frame_sequence_from_smb(smb_file_path, nb_frames, start_frame_idx=0):
	sequence = []
	for i in range(nb_frames):
		frame = get_frame_from_smb(smb_file_path, start_frame_idx+i)
		sequence.append(frame)

	sequence = np.array(sequence)
	#sequence = sequence[np.newaxis, :]
	return sequence


def predict_online(frames, nb_frames_per_sample, merge_weight=0.5, show_image=True):
	plt.figure(figsize=(3,3))
	img = None

	trim = frames.shape[0] - frames.shape[0]%nb_frames_per_sample
	frames = frames[:trim]
	sequences = frames.reshape((-1, 1, nb_frames_per_sample, img_size, img_size, 3))
	print(sequences.shape)

	y_preds = []

	start_time = time.time()

	# Go through all frames sequences one by one
	for i, sequence in enumerate(sequences):
		print("sequence {}/{}".format(i+1, len(sequences)), end='\r')
		im=sequence[-1, 0]
		im=np.multiply(sequence[-1, 0], -255).astype(np.uint8)#.copy() 

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
		if show_image:
			color = (255, 0, 0)
			cv2.putText(im, emotions[emo_pred], (2, 16), font, 0.7, color, 1, cv2.LINE_AA)
			if img is None:
				img = plt.imshow(im)
			else:
				img.set_data(im)

			plt.pause(.01)
			plt.draw()

	elapsed_time = time.time() - start_time
	avg_time_per_predictions = elapsed_time/len(sequences)

	print('')
	print('Finished online-predictions')
	print(' Average time per prediction : {:3f} seconds'.format(avg_time_per_predictions))


#### MAIN CODE ####

# sift_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/sift-phrnn1_TS4_DRIVE.h5'
lm_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/landmarks-phrnn1_TS4_DRIVE.h5'
tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/tcnn1_TS4_DRIVE.h5'
squeezenet_tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/squeezenet_tcnn1_TS4_DRIVE.h5'

# Prepare TCNN model
conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)
tcnn_top = create_tcnn_top()()
tcnn_top.load_weights(tcnn_model_path)

# Prepare PHRNN model
phrnn = create_phrnn_model(features_per_lm=2)()
phrnn.load_weights(lm_phrnn_model_path)

smb_file_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/smb/TS4_DRIVE/20180824_150225.smb'

# Get data and lauch online prediction
nb_frames = 100
frames = get_frame_sequence_from_smb(smb_file_path, nb_frames)
print(frames.shape)

predict_online(frames, nb_frames_per_sample=5)

