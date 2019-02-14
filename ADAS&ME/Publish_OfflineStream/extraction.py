import struct
import numpy as np
import os

from imageio import imread, imwrite
from scipy.misc import imresize
from imutils import face_utils
import dlib
import cv2

from const import *

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Reshape, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, concatenate, average
from keras.layers import Dense, Convolution2D, SimpleRNN, LSTM, Bidirectional
from keras import backend as K

### Extracting data from SMB files ###

def get_face_square(left, top, width, height, scale_factor):
	"""Returns the square around the face that should be used to crop the image.
	"""
	right = left+width
	bottom = top+height
	center_x = (left + right)/2
	center_y = (top + bottom)/2

	# Make the size of the square slightly bigger than in the ROI data
	square_len = scale_factor*max(width, height)

	half_len = square_len/2
	new_left = int(center_x - half_len)
	new_right = int(center_x + half_len)
	new_top = int(center_y - half_len)
	new_bottom = int(center_y + half_len)

	return ((new_left, new_top), (new_right, new_bottom))

def read_smb_header(file):
	'''Read SMB header:
	Jump to the position where the width and height of the image is stored in SMB header
	'''
	file.seek(12)
	image_width = bytearray(file.read(4))
	image_width.reverse()
	image_width = int(str(image_width).encode('hex'), 16)

	image_height = bytearray(file.read(4))
	image_height.reverse()
	image_height = int(str(image_height).encode('hex'), 16)

	return image_width, image_height

def read_roi_data(file):
	"""Read ROI data.
	"""
	camera_index = bytearray(file.read(4))
	camera_index.reverse()
	camera_index = int(str(camera_index).encode('hex'), 16)

	frame_number = bytearray(file.read(8))
	frame_number.reverse()
	frame_number = int(str(frame_number).encode('hex'), 16)

	time_stamp = bytearray(file.read(8))
	time_stamp.reverse()
	time_stamp = int(str(time_stamp).encode('hex'), 16)

	roi_left = bytearray(file.read(4))
	roi_left.reverse()
	roi_left = int(str(roi_left).encode('hex'), 16)

	roi_top = bytearray(file.read(4))
	roi_top.reverse()
	roi_top = int(str(roi_top).encode('hex'), 16)

	roi_width = bytearray(file.read(4))
	roi_width.reverse()
	roi_width = int(str(roi_width).encode('hex'), 16)

	roi_height = bytearray(file.read(4))
	roi_height.reverse()
	roi_height = int(str(roi_height).encode('hex'), 16)

	camera_angle = bytearray(file.read(8))
	camera_angle = struct.unpack('d', camera_angle)[0]

	return camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle



def read_image(file, current_position, roi_left, roi_top, roi_width, roi_height, image_width):

	file.seek(current_position + image_width * roi_top)
	image = bytearray(file.read(image_width * roi_height))
	image = np.array(image)

	
	if image.size == image_width * roi_height:
		image = np.reshape(image, (roi_height, image_width))
		image = image[:, roi_left : roi_left + roi_width]
	else:
		image = np.array([])

	return image


### Extracting Facial Landmarks from images ###

def detect_face_landmarks(gray_img, detector, predictor):
    """Detects 51 facial landmarks (eyes, eyebrows, nose, mouth) using dlib.using
    """
    # detect face in the grayscale image
    rects = detector(gray_img, 1)
    
    if len(rects)==0:
        # if no face was detected, we set the rectangle arbitrarily
        p = 24
        rect = dlib.rectangle(p,p,gray_img.shape[0]-p,gray_img.shape[1]-p)
    else:
        rect = rects[0]

    # determine the facial landmarks for the face region
    face_landmarks = face_utils.shape_to_np(predictor(gray_img, rect))
    face_landmarks = face_landmarks[17:] # remove landmarks around the face (useless)
    
    return face_landmarks

def extract_all_face_landmarks(x_data, verbose=True):
	"""Computes all the facial landmarks of every frame in the given dataset.
	"""
	# Initialize dlib's face detector and create the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(face_predictor_path)

	x_int = np.uint8(np.multiply(x_data, 256))
	face_landmarks = []

	for i, frame in enumerate(x_int):
		if verbose: 
			print("Computing facial landmarks... {:.1f}%".format(100.*(i+1)/x_int.shape[0]), '\r')
		if len(frame.shape)==2:
			gray=frame
		else:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		lm = detect_face_landmarks(gray, detector, predictor)
		face_landmarks.append(lm)
	
	if verbose: print('')
	face_landmarks = np.array(face_landmarks)
	
	return face_landmarks

def extract_landmarks_from_sequence(frame_sequence):
	# Prepare landmarks features for PHRNN
	nb_frames_per_sample = len(frame_sequence)

	landmarks = extract_all_face_landmarks(frame_sequence[:,0], verbose=False)
	lm_norm = normalize_face_landmarks(landmarks)

	eyebrows_landmarks = lm_norm[:, eyebrows_mask].reshape((1, nb_frames_per_sample, -1))
	nose_landmarks = lm_norm[:, nose_mask].reshape((1, nb_frames_per_sample, -1))
	eyes_landmarks = lm_norm[:, eyes_mask].reshape((1, nb_frames_per_sample, -1))
	inside_lip_landmarks = lm_norm[:, inside_lip_mask].reshape((1, nb_frames_per_sample, -1))
	outside_lip_landmarks = lm_norm[:, outside_lip_mask].reshape((1, nb_frames_per_sample, -1))

	x_landmarks = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]

	return x_landmarks
	
def normalize_face_landmarks(face_landmarks):
	"""Normalize facial landmarks by subtracting the coordinates corresponding
	   to the nose, and dividing by the standard deviation.
	"""
	face_landmarks_norm = np.zeros(face_landmarks.shape)
	
	for (i, lm) in enumerate(face_landmarks):
		face_landmarks_norm[i] = lm - lm[nose_center_idx]
			
	std_x = np.std(face_landmarks_norm[:,:,0].reshape((-1,)))
	std_y = np.std(face_landmarks_norm[:,:,1].reshape((-1,)))
	
	face_landmarks_norm[:,:,0] = np.multiply(face_landmarks_norm[:,:,0], 1./std_x)
	face_landmarks_norm[:,:,1] = np.multiply(face_landmarks_norm[:,:,1], 1./std_y)
	
	return face_landmarks_norm


### Extracting TCNN features ###

def get_conv_1_1_weights(vgg_weights_path):
	"""Returns the weigths of first convolutional layer of VGG-Face (conv_1_1).
	"""
	temp_mod = Sequential()
	temp_mod.add(ZeroPadding2D((1,1),input_shape=(224, 224, 3)))
	temp_mod.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
	temp_mod.load_weights(vgg_weights_path, by_name=True)
	conv1_1_weigths = temp_mod.get_layer('conv1_1').get_weights()
	return conv1_1_weigths

def create_tcnn_bottom(vgg_weights_path, conv1_1_weigths):
	"""Creates the bottom part of the VGG-16 TCNN.
	"""
	# Create inputs for the 5 frames
	input_shape=(224, 224, 3)
	frame1 = Input(shape=input_shape)
	frame2 = Input(shape=input_shape)
	frame3 = Input(shape=input_shape)
	frame4 = Input(shape=input_shape)
	frame5 = Input(shape=input_shape)
	
	# Convolution for each frame
	frame1_conv = ZeroPadding2D((1,1))(frame1)
	frame1_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1a')(frame1_conv)

	frame2_conv = ZeroPadding2D((1,1))(frame2)
	frame2_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1b')(frame2_conv)

	frame3_conv = ZeroPadding2D((1,1))(frame3)
	frame3_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1c')(frame3_conv)

	frame4_conv = ZeroPadding2D((1,1))(frame4)
	frame4_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1d')(frame4_conv)

	frame5_conv = ZeroPadding2D((1,1))(frame5)
	frame5_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1e')(frame5_conv)
	
	# Temporal aggregation by averaging
	temp_aggr = average([frame1_conv, frame2_conv, frame3_conv, frame4_conv, frame5_conv])

	# Then standard VGG-16 architecture
	output = ZeroPadding2D((1,1))(temp_aggr)
	output = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(output)
	output = MaxPooling2D((2,2), strides=(2,2))(output)

	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(128, (3, 3), activation='relu', name='conv2_1')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(128, (3, 3), activation='relu', name='conv2_2')(output)
	output = MaxPooling2D((2,2), strides=(2,2))(output)

	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(256, (3, 3), activation='relu', name='conv3_1')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(256, (3, 3), activation='relu', name='conv3_2')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(256, (3, 3), activation='relu', name='conv3_3')(output)
	output = MaxPooling2D((2,2), strides=(2,2))(output)

	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv4_1')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv4_2')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv4_3')(output)
	output = MaxPooling2D((2,2), strides=(2,2))(output)

	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv5_1')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv5_2')(output)
	output = ZeroPadding2D((1,1))(output)
	output = Convolution2D(512, (3, 3), activation='relu', name='conv5_3')(output)
	output = MaxPooling2D((2,2), strides=(2,2))(output)

	inputs = [frame1, frame2, frame3, frame4, frame5]
	model = Model(inputs=inputs, outputs=output)
	
	# load VGG-face weigths
	model.load_weights(vgg_weights_path, by_name=True)
	for layer in ['conv1_1a', 'conv1_1b', 'conv1_1c', 'conv1_1d', 'conv1_1e']:
		model.get_layer(layer).set_weights(conv1_1_weigths)

	return model

def fire_module(x, fire_id, squeeze=16, expand=64):
	"""Helper function for the creation of the Squeezenet model.
	"""
	sq1x1 = "squeeze1x1"
	exp1x1 = "expand1x1"
	exp3x3 = "expand3x3"
	relu = "relu_"
	s_id = 'fire' + str(fire_id) + '/'

	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3
	
	x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
	x = Activation('relu', name=s_id + relu + sq1x1)(x)

	left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
	left = Activation('relu', name=s_id + relu + exp1x1)(left)

	right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
	right = Activation('relu', name=s_id + relu + exp3x3)(right)

	x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
	return x

def get_conv1_weights(squeezenet_weights_path):
	"""Returns the weights of first convolutional layer of Squeezenet (conv_1).
	"""
	temp_mod = Sequential()
	temp_mod.add(Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1',input_shape=(227, 227, 3)))
	temp_mod.load_weights(squeezenet_weights_path, by_name=True)
	conv1_weights = temp_mod.get_layer('conv1').get_weights()
	return conv1_weights

def create_squeezenet_tcnn_bottom(squeezenet_weights_path, conv1_weights):
	"""Creates the bottom part of the Squeezenet TCNN.
	"""
	# Create inputs for the 5 frames
	input_shape=(227, 227, 3)
	frame1 = Input(shape=input_shape, name='frame1')
	frame2 = Input(shape=input_shape, name='frame2')
	frame3 = Input(shape=input_shape, name='frame3')
	frame4 = Input(shape=input_shape, name='frame4')
	frame5 = Input(shape=input_shape, name='frame5')
	
	# Convolution for each frame
	frame1_conv = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv1a')(frame1)
	frame2_conv = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv1b')(frame2)
	frame3_conv = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv1c')(frame3)
	frame4_conv = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv1d')(frame4)
	frame5_conv = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv1e')(frame5)
	
	# Temporal aggregation by averaging
	temp_aggr = average([frame1_conv, frame2_conv, frame3_conv, frame4_conv, frame5_conv])

	# Then standard Squeezenet architecture
	output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(temp_aggr)

	output = fire_module(output, fire_id=2, squeeze=16, expand=64)
	output = fire_module(output, fire_id=3, squeeze=16, expand=64)
	output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(output)

	output = fire_module(output, fire_id=4, squeeze=32, expand=128)
	output = fire_module(output, fire_id=5, squeeze=32, expand=128)
	output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(output)

	output = fire_module(output, fire_id=6, squeeze=48, expand=192)
	output = fire_module(output, fire_id=7, squeeze=48, expand=192)
	output = fire_module(output, fire_id=8, squeeze=64, expand=256)
	output = fire_module(output, fire_id=9, squeeze=64, expand=256)

	inputs = [frame1, frame2, frame3, frame4, frame5]
	model = Model(inputs=inputs, outputs=output)
	
	# load Squeezenet weights
	model.load_weights(squeezenet_weights_path, by_name=True)
	for layer in ['conv1a', 'conv1b', 'conv1c', 'conv1d', 'conv1e']:
		model.get_layer(layer).set_weights(conv1_weights)

	return model
