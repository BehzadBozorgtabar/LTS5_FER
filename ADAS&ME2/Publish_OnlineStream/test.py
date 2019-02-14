import sys, os, time, json
import paho.mqtt.client as mqtt
import numpy as np
import struct
import ctypes

import TCPStreamReader

from keras.models import load_model
from extract_smb_frames import *

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Some constant values
nb_channels = 3
emotions = {1: "Neutral", 2: "Positive", 3: "Frustrated", 4: "Anxiety"} #Emotions you want to predict, modify it to fit the ouput of the model used

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

model_path: the path of the model used for prediction
mqttHost: the host name or the ip address of the mqtt broker used for publication
client_name: the name of the client to make the connection with the mqtt broker
port: the port used for the tcp connection with the broker
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
Makes a TCP connection with a Smart Eye camera
Makes the predictions in streaming
Publishes the predictions to the MQTT broker

model: the model used for prediction
output_client: the client connected to the mqtt broker used to publish the prediction to the core pc
input_host: the ip address of the smart eye camera
width: the width of the image input
height: the height of the image input
img_size: the size of the images to fit the input of the model
type_algorithm: the algorithm used
topic: the topic on which we want to publish the predictions to the MQTT broker
"""
def prediction(model, output_client, input_host, input_port, width, height, img_size=224, type_algorithm="VGG_FACE", topic='emotion'):

	try:
		''' 
		Set up a TCP connection
		'''
		TCPStreamReader.initTcpStream(input_host, input_port, width, heigth)

		while True:
			img = TCPStreamReader.getImage(width, height) #Read the image on streaming
			if(img is not None):
				#Read the Roi Data
				camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(img)

				#Transform the image read on stream to a numpy array
				buf = (ctypes.c_uint8*(width*height)).from_address(int(img))
				frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)

				image = frame / 255.
			
				# Resize image to fit with the input of the model
				image = imresize(image, (img_size, img_size))
				image = image.reshape(1,img_size, img_size)
				image = np.stack([image]*nb_channels, axis = 3)

				predictions = model.predict(image).reshape(-1)
				predicted_class = np.argmax(predictions) + 1
				emotion = emotions[predicted_class]

				data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : int(predicted_class), "confidence" : "%.2f" % float(predictions[predicted_class-1]) }

				data_output = json.dumps(data)

				# Publication of the JSON object
				output_client.publish(topic, data_output)
			else:
				TCPStreamReader.free_roiData(img)
	finally:
		TCPStreamReader.deInitTcpStream()
		output_client.disconnect()
		print("Disconnected")

