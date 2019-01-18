import sys, os, time, json
import paho.mqtt.client as mqtt
import numpy as np

from keras.models import load_model
from extract_smb_frames import *

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Some constant values
nb_channels = 3
SMB_HEADER_SIZE = 20
emotions = {1: "Neutral", 2: "Positive", 3: "Frustrated", 4: "Anxiety"}

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
def init(model_path, mqttHost="localhost", client_name="JSON", port=1883):

	ktf.set_session(get_session())

	model = load_model(model_path)

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

	return model, client

"""
Make the predictions for all frames in the smb file.
It also published the output to the mqtt broker
"""
def prediction(model, filetest, client, img_size=224, type_algorithm="VGG_FACE", topic='emotion'):

	"""Extract the image frames from a specified smb file. 
	Face alignment and resizing is also done for testing.
	"""

	file = open(filetest,  "rb")
	try:
		# Read SMB header
		image_width, image_height = read_smb_header(file)
		current_position = 0
		file.seek(current_position)

		while True:
			# Read SMB header
			smb_header = file.read(SMB_HEADER_SIZE)
			if not smb_header:
				break


			# Read ROI data
			camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(file)

			# Read image
			file.seek(current_position + SMB_HEADER_SIZE)
			image = read_image(file, current_position, roi_left, roi_top, roi_width, roi_height, image_width)

			image = image / 255.
			
			# Resize image to fit with the input of the model
			image = imresize(image, (img_size, img_size))
			image = image.reshape(1,img_size, img_size)
			image = np.stack([image]*nb_channels, axis = 3)

			predictions = model.predict(image).reshape(-1)
			predicted_class = np.argmax(predictions) + 1
			emotion = emotions[predicted_class]

			data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : int(predicted_class), "confidence" : "%.2f" % float(predictions[predicted_class-1]) }

			data_output = json.dumps(data)

			client.publish(topic, data_output)

			# Jump to the next image
			current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
			file.seek(current_position)

	finally:
		file.close()		
		client.disconnect()
		print("Disconnected")
