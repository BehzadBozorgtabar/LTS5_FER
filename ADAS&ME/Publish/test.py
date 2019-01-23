import sys, os, time, json
import paho.mqtt.client as mqtt
import numpy as np

from keras.models import load_model
from extraction import *
from const import *

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
def prediction(tcnn_model, phrnn_model, tcnn_extractor, filetest, client, img_size=224, nb_frames_per_sample=5, tcnn_type="vgg", phrnn_type="landmarks", merge_weight=0.5, topic='emotion'):

	"""Extract the image frames from a specified smb file. 
	Face alignment and resizing is also done for testing.
	"""
	type_algorithm = 'late fusion ({}-tcnn, {}-phrnn)'.format(tcnn_type, phrnn_type)

	file = open(filetest,  "rb")
	try:
		# Read SMB header
		image_width, image_height = read_smb_header(file)
		current_position = 0
		file.seek(current_position)

		while True:
			frame_sequence = []
			for i in range(nb_frames_per_sample):
				# Read SMB header
				smb_header = file.read(SMB_HEADER_SIZE)
				if not smb_header:
					break

				# Read ROI data
				camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(file)

				# Read image
				file.seek(current_position + SMB_HEADER_SIZE)
				image = read_image(file, current_position, roi_left, roi_top, roi_width, roi_height, image_width)
				# image = image / 255.

				# Resize image to fit with the input of the model
				image = imresize(image, (img_size, img_size))
				image = image.reshape(1,img_size, img_size)
				image = np.stack([image]*nb_channels, axis = 3)
				frame_sequence.append(image)

				# Jump to the next image
				current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
				file.seek(current_position)

			frame_sequence = np.array(frame_sequence)
			tcnn_features = tcnn_extractor([frame_sequence[i] for i in range(nb_frames_per_sample)])
			landmarks = extract_landmarks_from_sequence(frame_sequence)

			# Compute predictions
			pred_tcnn = tcnn_model.predict(tcnn_features)
			pred_phrnn = phrnn_model.predict(landmarks)

			# Merging TCNN and PHRNN predictions
			predictions = merge_weight*pred_tcnn + (1-merge_weight)*pred_phrnn
			predicted_class = np.argmax(predictions) + 1
			emotion = emotions[predicted_class]

			print(emotion, predictions, pred_tcnn, pred_phrnn)
			data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : int(predicted_class), "confidence" : "%.2f" % float(predictions[predicted_class-1]) }
			data_output = json.dumps(data)

			client.publish(topic, data_output)

	finally:
		file.close()		
		client.disconnect()
		print("Disconnected")
