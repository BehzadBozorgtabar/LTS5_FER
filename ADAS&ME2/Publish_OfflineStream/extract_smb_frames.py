import struct
import numpy as np
import os

from imageio import imread, imwrite
from scipy.misc import imresize
import cv2


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
