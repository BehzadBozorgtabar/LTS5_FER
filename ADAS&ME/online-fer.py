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

def get_frame_sequence_from_smb(smb_file_path, start_frame_idx, nb_frames):
	sequence = []
	for i in range(nb_frames):
		frame = get_frame_from_smb(smb_file_path, start_frame_idx+i)
		sequence.append(frame)

	sequence = np.array(sequence)
	sequence = sequence[np.newaxis, :]
	return sequence


def predict_online(smb_file_path, nb_frames_per_sample, merge_weight=0.5):
	plt.figure(figsize=(3,3))
	img = None

	start_time = time.time()

	# Go through frames sequences one by one
	sequence_idx = 0
	while True:
		sequence = get_frame_sequence_from_smb(smb_file_path, sequence_idx*nb_frames_per_sample, nb_frames_per_sample)

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

		# Display image with predicted emotion on top
		color = (255, 0, 0)
		cv2.putText(im, emotions[emo_pred], (2, 16), font, 0.7, color, 1, cv2.LINE_AA)
		if img is None:
			img = plt.imshow(im, cmap='gray')
		else:
			img.set_data(im)

		plt.pause(.01)
		plt.draw()

		elapsed_time = time.time() - start_time
		avg_infer_time = elapsed_time/(sequence_idx+1)

		print('Sequence {} - Average inference time : {:3f} seconds'.format(sequence_idx, avg_infer_time))

		sequence_idx += 1


#### MAIN CODE ####

sift_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/sift-phrnn1_TS4_DRIVE.h5'
lm_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/landmarks-phrnn1_TS4_DRIVE.h5'
tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/tcnn1_TS4_DRIVE.h5'

# Prepare TCNN model
conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)
tcnn_top = create_tcnn_top()()
tcnn_top.load_weights(tcnn_model_path)

# Prepare PHRNN model
phrnn = create_phrnn_model(features_per_lm=2)()
phrnn.load_weights(lm_phrnn_model_path)

smb_file_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/smb/TS4_DRIVE/20180824_150225.smb'

# launch online prediction
predict_online(smb_file_path, nb_frames_per_sample=5)

