import sys, os, time, json
import struct
import ctypes
import numpy as np

import TCPStreamReader

from scipy.misc import imresize

from MQTT import on_connect, init_mqtt_connection, publish_output, disconnect_client
from FER import init_gpu_session, init_model, predict

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

"""
Allows to disconnect the computer from the smart eye
"""
def deInit_smart_eye_connection():
    return TCPStreamReader.deInitTcpStream()


"""
Extracts byte data from the stream input
"""
def swig_obj_unpack(b, offset, typ):
	num_bytes = struct.calcsize(typ)
	var = (ctypes.c_uint8*num_bytes).from_address(int(b)+offset)
	var = struct.unpack("<" + typ, bytearray(var))
	return var[0]

"""
Prints the Roi Header
"""
def printRoiHeader():
	print("CameraNumber\tFrameNumber\tTimeStamp\tTop\tLeft\tWidth\tHeight\tCameraAnglet")

"""
Prints the Roi values
Argument:
	camNum 		: The camera index
	frameNum 	: The frame number
	timeStamp 	: The timeStamp of the corresponding frame
	top 		: The bounding face box top_left corner y coordinate
	left 		: The bounding face box top_left corner x coordinate
	width		: The bounding face box width
	height		: The bounding face box height
	camAngle	: The angle of the camera
"""
def printRoiValue(camNum, frameNum, timeStamp,top,left,width,height,camAngle):
	print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(camNum,frameNum,timeStamp,left,top,width,height,camAngle))

"""
Allows to read the roi data from the img read online
Argument:
	img		: The image read from the TCP stream
Returns:
	Roi values (Same as in printRoiValue method)
"""
def read_roi_data_online(img):
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
Allows to read an image in streaming with the smart eye camera
Arguments:
    width   : The width of the image we want
    heigth  : The height of the image we want
Returns:
    A stream buffer containing the image data
"""
def getImage(width, height):
    return TCPStreamReader.getImage(width, height)

"""
Allows to free the ROiData from memory
Argument:
    img : the stream buffer image
"""
def freeRoiData(img):
    return TCPStreamReader.free_roiData(img)

"""
Allows to convert an image read on stream with getImage method to a numpy array
Arguments:
    img     : The stream buffer image
    width   : The width of the image we want
    height  : The height of the image we want
"""
def stream_to_numpy(img, width, height):
    buf = (ctypes.c_uint8*(width*height)).from_address(int(img))
    frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
    return frame

###################################### ALGORITHM EXAMPLE FOR ONLINE DETECTION (NOT TESTED) ######################################

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

	init_gpu_session()

	if type_model == "vgg":
		type_algorithm = "Custom_VGG_FACE"
		img_size = 224
		nb_channels = 3
	elif type_model == "squeezenet":
		type_algorithm = "Custom_Squeeze_Net"
		img_size = 227
		nb_channels = 3
	else:
		print("Invalid type model, choose between vgg and squeezenet please.")
		sys.exit()

	model = init_model(model_name) #Loads the model, now ready to make the predictions
	output_client = init_mqtt_connection(mqttHost, client_name, port) #Create a client connected to a MQTT broker to publish the outputs

	try:
		''' 
		Set up a TCP connection
		'''
		init_smart_eye_connection(input_host, input_port, width, height)

		while True:
			img = getImage(width, height) #Read the image on streaming
			if(img is not None):
				#Read the Roi Data
				camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data_online(img)

				#Transform the image read on stream to a numpy array
				image = stream_to_numpy(img, width, height)

				#Resize image to fit with the input of the model
				image = image[roi_top : roi_top + roi_height, roi_left : roi_left + roi_width]
				image = image / 255.

				image = imresize(image, (img_size, img_size))
				image = image.reshape(1,img_size, img_size)
				image = np.stack([image]*nb_channels, axis = 3)

				#Prediction part
				emotion, predicted_class, confidence, predictions = predict(model, image)

				data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : predicted_class, "confidence" : confidence }

				publish_output(output_client, data, topic)

			else:
				freeRoiData(img)
	finally:
		deInit_smart_eye_connection()
		disconnect_client(output_client)
		print("Disconnected")
