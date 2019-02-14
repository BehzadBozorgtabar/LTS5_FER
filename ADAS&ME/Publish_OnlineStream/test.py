import sys, os, time, json
import paho.mqtt.client as mqtt
import numpy as np
import struct
import ctypes

import TCPStreamReader

from keras.models import load_model
from extraction import *
from const import *

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""
Prints the Roi Header
"""
def printRoiHeader():
	print("CameraNumber\tFrameNumber\tTimeStamp\tTop\tLeft\tWidth\tHeight\tCameraAnglet")

"""
Prints the Roi values
"""
def printRoiValue(camNum, frameNum, timeStamp,top,left,width,height,camAngle):
	print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(camNum,frameNum,timeStamp,left,top,width,height,camAngle))

"""
Extracts byte data from the stream input
"""
def swig_obj_unpack(b, offset, typ):
	num_bytes = struct.calcsize(typ)
	var = (ctypes.c_uint8*num_bytes).from_address(int(b)+offset)
	var = struct.unpack("<" + typ, bytearray(var))
	return var[0]

"""
Allows to read the roi data from the img
"""
def read_roi_data(img):
	camera_num = swig_obj_unpack(img,  0, 'I')
	frame_num = swig_obj_unpack(img, 4, 'Q')
	timeStamp = swig_obj_unpack(img, 12, 'Q')
	boundingbox_left = swig_obj_unpack(img, 20, 'I')
	boundingbox_top = swig_obj_unpack(img, 24, 'I')
	boundingbox_width = swig_obj_unpack(img, 28, 'I')
	boundingbox_height = swig_obj_unpack(img, 32, 'I')
	camer_angle = swig_obj_unpack(img, 36, 'd')
	return camera_num, frame_num, timeStamp, boundingbox_left, boundingbox_top, boundingbox_width, boundingbox_height, camer_angle

"""
Callback function when connected to the MQTT
"""
def on_connect(client, userdata, flags, rc):
	if rc == 0:
		client.connected_flag = True
		print("Connection successful")
	else:
		print("Connection unsuccessful, returned code = ",rc)
		sys.exit()

"""
Init function, returns the model and the client connected to the MQTT broker
"""
def init(mqttHost="localhost", client_name="JSON", port=1883):

	ktf.set_session(get_session())

	### CONNECTION TO MQTT BROKER ###
	mqtt.Client.connected_flag = False

	client = mqtt.Client(client_name)
	client.on_connect = on_connect
	client.loop_start()
	print("Connecting to broker ", mqttHost)
	client.connect(mqttHost, port = port)
	while not client.connected_flag:
		print("Waiting...")
		time.sleep(1)
	print("Waiting...")
	client.loop_stop()
	print("Starting main program")

	return client


"""
Make the predictions for all frames in the smb file.
It also published the output to the mqtt broker
"""
def prediction(tcnn_model, phrnn_model, tcnn_extractor, output_client, input_host, input_port, width, height, img_size=224, nb_frames_per_sample=5, tcnn_type="vgg", phrnn_type="landmarks", merge_weight=0.5, topic='emotion'):

	"""Extract the image frames from a specified smb file. 
	Face alignment and resizing is also done for testing.
	"""
	type_algorithm = 'late fusion ({}-tcnn, {}-phrnn)'.format(tcnn_type, phrnn_type)

	try:
		''' 
		Set up a TCP connection
		'''
		TCPStreamReader.initTcpStream(input_host, input_port, width, heigth)

		while True:
			frame_sequence = []
			for i in range(nb_frames_per_sample):
				img = TCPStreamReader.getImage(width, height);
				if(img is not None):
					camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(img)

					buf = (ctypes.c_uint8*(width*height)).from_address(int(img))
					image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)

					# Resize image to fit with the input of the model
					image = imresize(image, (img_size, img_size))
					image = image.reshape(1,img_size, img_size)
					image = np.stack([image]*nb_channels, axis = 3)
					frame_sequence.append(image)
				else:
					TCPStreamReader.free_roiData(img)

			frame_sequence = np.array(frame_sequence)
			tcnn_features = tcnn_extractor([frame_sequence[i] for i in range(nb_frames_per_sample)])
			landmarks = extract_landmarks_from_sequence(frame_sequence)

			# Compute predictions
			pred_tcnn = tcnn_model.predict(tcnn_features)
			pred_phrnn = phrnn_model.predict(landmarks)

			# Merging TCNN and PHRNN predictions
			predictions = (merge_weight*pred_tcnn + (1-merge_weight)*pred_phrnn)[0]
			predicted_class = np.argmax(predictions) + 1
			emotion = emotions[predicted_class]

			data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : int(predicted_class), "confidence" : "%.2f" % float(predictions[predicted_class-1]) }
			print(data)
			data_output = json.dumps(data)

			output_client.publish(topic, data_output)

	finally:
		TCPStreamReader.deInitTcpStream()
		output_client.disconnect()
		print("Disconnected")
