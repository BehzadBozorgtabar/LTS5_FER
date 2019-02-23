import sys, os, time, json
import paho.mqtt.client as mqtt
import numpy as np
import ctypes

import TCPStreamReader
from extract_smb_frames import read_smb_header, read_roi_data, read_image
from extract_online_frames import read_roi_data_online, printRoiHeader, printRoiValue

from keras.models import load_model

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf



############################################################################################
#######################################  INIT PART  ########################################

"""
Loads the model used for prediction
Arguments:
	model_path	: The path of the model used for predictions
Returns:
	The loaded model
"""
def init_model(model_path):
	return load_model(model_path)

"""
Callback function when connected to the MQTT
This function is automatically called during the connection step, you don't need to use it
"""
def on_connect(client, userdata, flags, rc):
	if rc == 0:
		client.connected_flag = True
		print("Connection successful")
	else:
		print("Connection unsuccessful, returned code = ",rc)
		sys.exit()

"""
Connects the client to the MQTT broker for the publication step
Arguments:
	mqttHost	: The host name or the ip address of the mqtt broker used for publication
	client_name	: The name of the client to make the connection with the mqtt broker
	port		: The port used for the tcp connection with the broker
Returns:
	The client connected to the MQTT broker
"""
def init_mqtt_connection(mqttHost, client_name, port):
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
Setups the jetson GPU to avoid memory error during the predictions
"""
def init_gpu_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	ktf.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

"""
Connects the computer to the smart eye camera with a TCP connection
Arguments:
	host		: The ip address of the smart eye camera
	port		: The port used to make the connection
	width		: The width of the image input
	height		: The height of the image input
"""
def init_smart_eye_connection(host, port, width, height):
	TCPStreamReader.initTcpStream(input_host, input_port, width, heigth)



############################################################################################
#####################################  ALGORITHM PART  #####################################

"""
Makes the prediction for the input image.
Arguments:
	- model 	: The loaded model used for prediction
	- image 	: The input image, it has to be a 1*224*224*3 numpy array. Plus the values have to be between 0 and 1 instead of 0 and 255
	- emotions 	: A dictionnary containing the possible predictions
Returns:
	- emotion 	: The emotion name predicted (String)
	- predicted_class : The corresponding index of the prediction (Int)
	- confidence	: The confidence of the prediction (String)
"""
def predict(model, image, emotions = {1: "Neutral", 2: "Positive", 3: "Frustrated", 4: "Anxiety"}):
	prediction = model.predict(image).reshape(-1)
	predicted_class = np.argmax(prediction) + 1
	emotion = emotions[predicted_class]
	confidence = "%.2f" % float(predictions[predicted_class-1])
	return emotion, int(predicted_class), confidence



############################################################################################
####################################  PUBLICATION PART  ####################################

"""
The client connected to the MQTT broker publishes the output on the topic "topic"
Arguments:
	- client	: The client connected to the MQTT broker
	- output	: The output of the algorithm
	- topic		: The topic on which we want to publish the output to the MQTT broker		
"""
def publish_output(client, output, topic):
	output = json.dumps(output)
	client.publish(topic, output)

"""
Disconnects the client connected to the MQTT broker
Argument:
	- client	: The client connected to the MQTT broker
"""
def disconnect_client(client):
	client.disconnect()

############################################################################################
#####################################  FULL PIPELINE  ######################################

"""
These are examples of implementation of the full pipeline.
The first is online prediction with a smart eye camera.
The second one is offline prediction on a SMB binary file.
"""


"""
Makes a TCP connection with a Smart Eye camera
Makes the predictions in streaming
Publishes the predictions to the MQTT broker

Arguments:
	model_name	: The model path used for prediction
	type_model	: The model type used for prediction ("vgg" or "squeezenet")
	input_host	: The ip address of the smart eye camera
	input_port	: The port used to make the connection
	width		: The width of the image input
	height		: The height of the image input
	topic		: The topic on which we want to publish the predictions to the MQTT broker
"""
def online_prediction(model_name, type_model, input_host, input_port, width, height, mqttHost="localhost", client_name="JSON", port=1883, topic='emotion'):

	if type_model == "vgg":
		type_algorithm = "Custom_VGG_FACE"
		img_size = 224
		nb_channels = 3
	elif type_model == "squeezenet":
		type_algorithm = "Custom_Squeeze_Net"
		img_size = 227
		nb_channels = 3
	else:
		print("Invalid type model, choose between vgg and squeezenet please."
		sys.exit()

	model = init_model(model_name) #Loads the model, now ready to make the predictions
	output_client = init_mqtt_connection(mqttHost, client_name, port) #Create a client connected to a MQTT broker to publish the outputs
	init_gpu_session()

	try:
		''' 
		Set up a TCP connection
		'''
		init_smart_eye_connection(input_host, input_port, width, height)

		while True:
			img = TCPStreamReader.getImage(width, height) #Read the image on streaming
			if(img is not None):
				#Read the Roi Data
				camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data_online(img)

				#Transform the image read on stream to a numpy array
				buf = (ctypes.c_uint8*(width*height)).from_address(int(img))
				frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)

				#Resize image to fit with the input of the model
				image = frame / 255.
			
				image = imresize(image, (img_size, img_size))
				image = image.reshape(1,img_size, img_size)
				image = np.stack([image]*nb_channels, axis = 3)

				#Prediction part
				emotion, predicted_class, confidence = predict(model, image)

				data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : predicted_class, "confidence" : confidence }

				publish_output(output_client, data, topic)

			else:
				TCPStreamReader.free_roiData(img)
	finally:
		TCPStreamReader.deInitTcpStream()
		disconnect_client(output_client)
		print("Disconnected")


"""
Make the predictions for all frames in the smb file.
It also publishes the output to the mqtt broker.

Arguments:
	model_name	: The model path used for prediction
	type_model	: The model type used for prediction ("vgg" or "squeezenet")
	filetest	: The SMB file path used for predictions
	topic		: The topic on which we want to publish the predictions to the MQTT broker
	SMB_HEADER_SIZE	: THE SMB HEADER SIZE of the SMB file
"""
def offline_prediction(model_name, type_model, filetest, mqttHost="localhost", client_name="JSON", port=1883, topic='emotion', SMB_HEADER_SIZE = 20):

	if type_model == "vgg":
		type_algorithm = "Custom_VGG_FACE"
		img_size = 224
		nb_channels = 3
	elif type_model == "squeezenet":
		type_algorithm = "Custom_Squeeze_Net"
		img_size = 227
		nb_channels = 3
	else:
		print("Invalid type model, choose between vgg and squeezenet please."
		sys.exit()

	model = init_model(model_name) #Loads the model, now ready to make the predictions
	output_client = init_mqtt_connection(mqttHost, client_name, port) #Create a client connected to a MQTT broker to publish the outputs
	init_gpu_session()

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

			# Resize image to fit with the input of the model
			image = image / 255.
			
			image = imresize(image, (img_size, img_size))
			image = image.reshape(1,img_size, img_size)
			image = np.stack([image]*nb_channels, axis = 3)

			emotion, predicted_class, confidence = predict(model, image)

			data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : predicted_class, "confidence" : confidence }

			publish_output(client, data, topic)

			# Jump to the next image
			current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
			file.seek(current_position)

	finally:
		file.close()		
		disconnect_client(client)
		print("Disconnected")

