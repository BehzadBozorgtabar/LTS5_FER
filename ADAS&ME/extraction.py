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



def extract_smb_frames(smb_data_path, smb_files_list, face_rect_scale=1.25, dest_folder=None, align_resize=True, img_resize=224, nb_frames=None):
    """Extract the image frames from the specified smb files. 
       Face alignment and resizing is also done to prepare for training.
    """
    if dest_folder is not None and not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    frames = []
    for smb_file in smb_files_list:
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
                print("Extracting from {} - frame {}".format(smb_file+'.smb', i+1), end='\r') 

                # Read SMB header
                smb_header = file.read(SMB_HEADER_SIZE)
                if not smb_header:
                    break

                # Read ROI data
                camera_index, frame_number, time_stamp, roi_left, roi_top, roi_width, roi_height, camera_angle = read_roi_data(file)

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
        finally:
            file.close()
            print('')
            #frames.append(subj_frames)
    
    #return np.array(frames)


### Extract data from preprocessed frames ###

def extract_data(frames_path, annotations_path):
    """Extracts trainingand target data (x & y) from the video frames.
    
    Keyword arguments:
    frames_path -- path of the folder in which the selected frames should be stored
    """
    x = []
    
    annotations = pd.read_csv(annotations_path)
    y = annotations.apply(lambda row: get_emotion_class(row['Valence'], row['Arousal']), axis=1).values

    frames = sorted([f for f in os.listdir(frames_path) if f.endswith('.png')])
    frames_ref = []

    for (i, frame) in enumerate(frames[:1000]):
        img = imread(frames_path+"/"+frame)

        # convert to rgb if greyscale
        if len(img.shape)==2:
            img = np.stack((img,)*3, axis=-1)

        x.append(img)
        frames_ref.append(int(frame[:-4]))
            
    x = np.multiply(np.array(x), 1/256)
    y = np.asarray(y[:x.shape[0]])
    
    return x, y, np.asarray(frames_ref)

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


def extract_data(frames_path, annotations_path):
    """Extracts trainingand target data (x & y) from the video frames.
    
    Keyword arguments:
    frames_path -- path of the folder in which the selected frames should be stored
    """
    x = []
    
    annotations = pd.read_csv(annotations_path)
    y = annotations.apply(lambda row: get_emotion_class(row['Valence'], row['Arousal']), axis=1).values

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
    y = np.asarray(y)
    
    return x, y, np.asarray(frames_ref)


def extract_sift_vgg(sift_extractor, vgg_extractor, frames_path, annotations_path, smb_files_list, dest_folder):
    """Extract and save vgg and sift features from the frames.
    """
    for smb_file in smb_files_list:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        print('Extracting from '+smb_file+'...')
        frames_path_full = frames_path+'/'+smb_file
        annotations_path_full = annotations_path+'/'+smb_file+'_annotated.csv'
        x, y, frames_ref = extract_data(frames_path_full, annotations_path_full)
        
        print(' SIFT features...')
        sift_features = sift_extractor(x)
        with open('{}/sift_{}.pkl'.format(dest_folder, smb_file), 'wb') as f:
            pickle.dump(sift_features, f)
        sift_features = None
        
        print(' VGG features...')
        vgg_features = vgg_extractor(x)
        with open('{}/vggCustom_{}.pkl'.format(dest_folder, smb_file), 'wb') as f:
            pickle.dump(vgg_features, f)
        vgg_features = None
        
        x = None

