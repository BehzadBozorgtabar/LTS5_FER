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
	image_width = int.from_bytes(image_width, byteorder='big')

	image_height = bytearray(file.read(4))
	image_height.reverse()
	image_height = int.from_bytes(image_height, byteorder='big')

	return image_width, image_height

def read_roi_data(file):
	"""Read ROI data.
	"""
	camera_index = bytearray(file.read(4))
	camera_index.reverse()
	camera_index = int.from_bytes(camera_index, byteorder='big')

	frame_number = bytearray(file.read(8))
	frame_number.reverse()
	frame_number = int.from_bytes(frame_number, byteorder='big')

	time_stamp = bytearray(file.read(8))
	time_stamp.reverse()
	time_stamp = int.from_bytes(time_stamp, byteorder='big')

	roi_left = bytearray(file.read(4))
	roi_left.reverse()
	roi_left = int.from_bytes(roi_left, byteorder='big')

	roi_top = bytearray(file.read(4))
	roi_top.reverse()
	roi_top = int.from_bytes(roi_top, byteorder='big')

	roi_width = bytearray(file.read(4))
	roi_width.reverse()
	roi_width = int.from_bytes(roi_width, byteorder='big')

	roi_height = bytearray(file.read(4))
	roi_height.reverse()
	roi_height = int.from_bytes(roi_height, byteorder='big')

	camera_angle = bytearray(file.read(8))
	camera_angle = struct.unpack('d', camera_angle)[0]

	return camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle

def read_image(file, image_width, image_height):
	"""Read the image from the SMB file.
	"""
	image = bytearray(file.read(image_width * image_height))
	image = np.array(image)
	if image.size == image_width * image_height:
		image = np.reshape(image, (image_height, image_width))
	else:
		image = np.array([])

	return image

def extract_smb_frames(smb_data_path, face_rect_scale=1.25, dest_folder=None, align_resize=True, img_resize=224, start_frame_index = 0, nb_frames=None, frames_nbr=None, verbose=True):
	"""Extract the image frames from a specified smb file. 
	Face alignment and resizing is also done to prepare for training.
	"""
	if dest_folder is not None and not os.path.exists(dest_folder):
		os.makedirs(dest_folder)

	SMB_HEADER_SIZE = 20
	frames = []
	file = open(smb_data_path, "rb")
	i = 0
	try:
		# Read SMB header
		image_width, image_height = read_smb_header(file)
		current_position = start_frame_index * (SMB_HEADER_SIZE + image_width*image_height)
		file.seek(current_position)

		while True:
			if verbose:
				print("Extracting frame {}".format(i+1), end = '\r')

			# Read SMB header
			smb_header = file.read(SMB_HEADER_SIZE)
			if not smb_header:
				break

			# Read ROI data
			camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(file)
			if frames_nbr is None or frame_number in frames_nbr:
				# Read image
				file.seek(current_position + SMB_HEADER_SIZE)
				image = read_image(file, image_width, image_height)
			    
				if align_resize:
					# Align image with face using roi data
					((l,t),(r,b)) = get_face_square(roi_left, roi_top, roi_width, roi_height, face_rect_scale)
					p_val = int(np.median(image))
					p = image.shape[1]//3
					img = np.array(cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=[p_val,p_val,p_val]))
					image = img[p+t:p+b,p+l:p+r]
				
				# Resize image to 224x224
				image = imresize(image, (img_resize, img_resize))
				
				if dest_folder is not None:
					# Write image in destination folder
					dest_path = '{}/{}.png'.format(dest_folder, frame_number)
					imwrite(dest_path, image)
				frames.append(image)

			# Jump to the next image
			current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
			file.seek(current_position)
			    
			i += 1
			if nb_frames is not None and (i >= nb_frames):
				break

	finally:
		file.close()

	return np.array(frames)		
