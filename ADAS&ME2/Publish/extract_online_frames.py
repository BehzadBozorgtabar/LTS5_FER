import struct
import ctypes

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
