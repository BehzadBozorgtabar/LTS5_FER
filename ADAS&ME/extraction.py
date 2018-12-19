import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from scipy.misc import imresize
import struct
import pandas as pd
import cv2
import pickle

from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Reshape, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation, concatenate, average
from keras.layers import Dense, Convolution2D, SimpleRNN, LSTM, Bidirectional

from sift import extract_all_face_landmarks
from training import get_file_annotations, get_labels_from_annotations
from const import *

### Extracting data from SMB files ###

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

def extract_smb_frames(smb_data_path, smb_files_list, face_rect_scale=1.25, dest_folder=None, align_resize=True, img_resize=224, nb_frames=None, stop_at_frame_nb=[]):
    """Extract the image frames from the specified smb files. 
       Face alignment and resizing is also done to prepare for training.
    """

    if dest_folder is not None and not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    frames = []
    for file_idx, smb_file in enumerate(smb_files_list):
        smb_file_path = smb_data_path+'/'+smb_file+'.smb'
        file = open(smb_file_path, "rb")
        SMB_HEADER_SIZE = 20
        subj_frames = []
        i = 0
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

                print("Extracting from {} - frame nb {} - frame id {}".format(smb_file+'.smb', i+1, frame_number), end='\r') 

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
                    sub_folder = dest_folder + '/' + smb_file
                    if not os.path.exists(sub_folder):
                        os.makedirs(sub_folder)
                        
                    # Write image in destination folder
                    dest_path = '{}/{}.png'.format(sub_folder, frame_number)
                    imwrite(dest_path, image)
                subj_frames.append(image)

                # Jump to the next image
                current_position = current_position + (image_width * image_height) + SMB_HEADER_SIZE
                file.seek(current_position)

                i += 1
                if nb_frames is not None and (i >= nb_frames):
                    break
                if len(stop_at_frame_nb)>0 and stop_at_frame_nb[file_idx] is not None and (stop_at_frame_nb[file_idx] == frame_number):
                    break
        finally:
            file.close()
            print('')
            #frames.append(subj_frames)
    
    #return np.array(frames)


### Extract data from preprocessed frames ###

def extract_data(frames_path):
    """Extracts trainingand target data (x & y) from the video frames.
    
    Keyword arguments:
    frames_path -- path of the folder in which the selected frames should be stored
    """
    x = []

    frames = sorted([f for f in os.listdir(frames_path) if f.endswith('.png')])
    frames_ref = []

    for (i, frame) in enumerate(frames):
        img = imread(frames_path+"/"+frame)

        # convert to rgb if greyscale
        if len(img.shape)==2:
            img = np.stack((img,)*3, axis=-1)

        x.append(img)
        frames_ref.append(int(frame[:-4]))
            
    x = np.multiply(np.array(x), 1/256)
    
    return x, np.asarray(frames_ref)

def extract_pickle_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        x, roi =  pickle.load(f)
        
    # convert frames to rgb if grayscale
    if len(x[0].shape)==2:
        x = np.stack((x,)*3, axis=-1)
        
    labels = [r['Severity']-1 for r in roi]
    camera_idx = [r['CameraIndex'] for r in roi]
    frames_idx = [r['FrameNumber'] for r in roi]
    
    x = np.multiply(np.array(x), 1/256)
    return x, labels, frames_idx

def create_vgg_extractor(model_path, output_layer='fc6'):
    """Loads the given VGG model and set the given layer as output
    """
    # Load the pre-trained VGG-face weights
    vgg = load_model(model_path)

    for layer in vgg.layers:
        layer.trainable = False

    # We want to take the ouput of the fc7 layer as input features of our next network
    fc_output = vgg.get_layer(output_layer).output
    vgg_fc = Model(inputs=vgg.input, outputs=fc_output)
    
    return vgg_fc

def extract_sift(sift_extractor, subj_data_path, dest_folder, verbose=True):
    """Extract and save vgg and sift features from the frames.
    """
    pickle_files = sorted([f for f in os.listdir(subj_data_path) if f.endswith('.pkl')])
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in pickle_files:
        if verbose: print('Extracting from '+file+'...')
        pickle_data_path_full = subj_data_path+'/'+file
        x, _ = extract_pickle_data(pickle_data_path_full)
        
        if verbose: print(' SIFT features...')
        sift_features = sift_extractor(x)
        print(" "+str(sift_features.shape))
        
        dest = dest_folder+'/sift'
        if not os.path.exists(dest):
            os.makedirs(dest)

        with open('{}/sift_{}.pkl'.format(dest, file[:-4]), 'wb') as f:
            pickle.dump(sift_features, f)
        sift_features = None
        x = None

def extract_sift_vgg(sift_extractor, vgg_extractor, subj_data_path, dest_folder, verbose=True):
    """Extract and save sift features from the frames.
    """
    pickle_files = sorted([f for f in os.listdir(subj_data_path) if f.endswith('.pkl')])
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in pickle_files:
        if verbose: print('Extracting from '+file+'...')
        pickle_data_path_full = subj_data_path+'/'+file
        x, _ = extract_pickle_data(pickle_data_path_full)
        
        if verbose: print(' SIFT features...')
        sift_features = sift_extractor(x)
        print(" "+str(sift_features.shape))
        
        dest = dest_folder+'/sift'
        if not os.path.exists(dest):
            os.makedirs(dest)

        with open('{}/sift_{}.pkl'.format(dest, file[:-4]), 'wb') as f:
            pickle.dump(sift_features, f)
        sift_features = None

        if verbose: print(' VGG features...')
        vgg_features = vgg_extractor(x)

        dest = dest_folder+'/vgg'
        if not os.path.exists(dest):
            os.makedirs(dest)

        with open('{}/vggCustom_{}.pkl'.format(dest_folder, file), 'wb') as f:
            pickle.dump(vgg_features, f)
        vgg_features = None
        x = None

def extract_facial_landmarks(subj_data_path, dest_folder, verbose=True):
    """Extract and save facial landmarks from the frames.
    """
    pickle_files = sorted([f for f in os.listdir(subj_data_path) if f.endswith('.pkl')])
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in pickle_files:
        if verbose: print('Extracting from '+file+'...')
        pickle_data_path_full = subj_data_path+'/'+file
        x, _ = extract_pickle_data(pickle_data_path_full)
        
        if verbose: print(' Facial landmarks features...')
        landmarks = extract_all_face_landmarks(x)
        print(" "+str(landmarks.shape))

        dest = dest_folder+'/landmarks'
        if not os.path.exists(dest):
            os.makedirs(dest)

        with open('{}/landmarks_{}.pkl'.format(dest, file[:-4]), 'wb') as f:
            pickle.dump(landmarks, f)
        landmarks = None
        x = None

def extract_vgg_tcnn(vgg_predictor, nb_frames_per_sample, subj_data_path, dest_folder, verbose=True):
    """Extract VGG-TCNN features from the frames.
    """
    pickle_files = sorted([f for f in os.listdir(subj_data_path) if f.endswith('.pkl')])
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in pickle_files:
        if verbose: print('Extracting from '+file+'...')
        pickle_data_path_full = subj_data_path+'/'+file
        x, _ = extract_pickle_data(pickle_data_path_full)

        # Group by sequences of 'nb_frames_per_sample' frames
        trim = x.shape[0] - x.shape[0]%nb_frames_per_sample
        x = x[:trim]
        x = x.reshape((-1, nb_frames_per_sample, img_size, img_size, 3))
        
        if verbose: print(' Computing VGG-TCNN features...')
        vgg_tcnn_features = vgg_predictor([x[:,i] for i in range(nb_frames_per_sample)])
        print(" "+str(vgg_tcnn_features.shape))
        
        dest = dest_folder+'/vgg_tcnn'
        if not os.path.exists(dest):
            os.makedirs(dest)
            
        with open('{}/vgg_tcnn_{}.pkl'.format(dest, file[:-4]), 'wb') as f:
            pickle.dump(vgg_tcnn_features, f)
        vgg_tcnn_features = None
        x = None

def get_conv_1_1_weights(vgg_weights_path):
    """Returns the weigths of first convolutional layer of VGG-Face (conv_1_1).
    """
    temp_mod = Sequential()
    temp_mod.add(ZeroPadding2D((1,1),input_shape=(img_size, img_size, 3)))
    temp_mod.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    temp_mod.load_weights(vgg_weights_path, by_name=True)
    conv1_1_weigths = temp_mod.get_layer('conv1_1').get_weights()
    return conv1_1_weigths

def create_tcnn_bottom(vgg_weights_path, conv1_1_weigths):
    """Creates the bottom part of the VGG-16 TCNN.
    """
    # Create inputs for the 5 frames
    input_shape=(img_size, img_size, 3)
    frame1 = Input(shape=input_shape)
    frame2 = Input(shape=input_shape)
    frame3 = Input(shape=input_shape)
    frame4 = Input(shape=input_shape)
    frame5 = Input(shape=input_shape)
    
    # Convolution for each frame
    frame1_conv = ZeroPadding2D((1,1))(frame1)
    frame1_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1a')(frame1_conv)

    frame2_conv = ZeroPadding2D((1,1))(frame2)
    frame2_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1b')(frame2_conv)

    frame3_conv = ZeroPadding2D((1,1))(frame3)
    frame3_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1c')(frame3_conv)

    frame4_conv = ZeroPadding2D((1,1))(frame4)
    frame4_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1d')(frame4_conv)

    frame5_conv = ZeroPadding2D((1,1))(frame5)
    frame5_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1e')(frame5_conv)
    
    # Temporal aggregation by averaging
    temp_aggr = average([frame1_conv, frame2_conv, frame3_conv, frame4_conv, frame5_conv])

    # Then standard VGG-16 architecture
    output = ZeroPadding2D((1,1))(temp_aggr)
    output = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(128, (3, 3), activation='relu', name='conv2_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(128, (3, 3), activation='relu', name='conv2_2')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    inputs = [frame1, frame2, frame3, frame4, frame5]
    model = Model(inputs=inputs, outputs=output)
    
    # load VGG-face weigths
    model.load_weights(vgg_weights_path, by_name=True)
    for layer in ['conv1_1a', 'conv1_1b', 'conv1_1c', 'conv1_1d', 'conv1_1e']:
        model.get_layer(layer).set_weights(conv1_1_weigths)

    return model
