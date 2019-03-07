import sys, os, time, json
import struct
import numpy as np

from scipy.misc import imresize

from MQTT import on_connect, init_mqtt_connection, publish_output, disconnect_client
from FER import init_gpu_session, init_model, predict


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

###################################### ALGORITHM EXAMPLE FOR ONLINE DETECTION (TESTED, IT WORKS) ######################################

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

			emotion, predicted_class, confidence, predictions = predict(model, image)

			data = { "timeStamp" : str(time_stamp), "statusCode" : 200, "type" : type_algorithm, "state" : emotion, "level" : predicted_class, "confidence" : confidence }

			publish_output(output_client, data, topic)

			# Jump to the next image
			current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
			file.seek(current_position)

	finally:
		file.close()		
		disconnect_client(output_client)
		print("Disconnected")
