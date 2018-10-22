import os
import numpy as np
from imageio import imread, imsave
from scipy.misc import imresize
from imutils import face_utils
import dlib
import cv2
from keras.models import Model, load_model

from const import *

#### RAW DATA EXTRACTION ####

def extract_data(img_path, label_path):
    """Extracts training and target data (x & y) from the raw data.
    
    Keyword arguments:
    frames_path -- path of the folder in which the selected frames should be stored
    """
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
    label_file = open(label_path, 'r')
    
    label_file = open(label_path, "r")

    lines = label_file.readlines()
    nb_imgs = len(lines)
    for (i, line) in enumerate(lines):
        img_name = line.split(' ')[0][:-4]+'_aligned.jpg'
        emotion_idx = int(line.split(' ')[1][0])-1
        
        # Read and resize image
        img = imread(img_path+"/"+img_name)
        img = imresize(img, (img_size, img_size))
        
        if img_name.startswith('train'):
            x_train.append(np.array(img))
            y_train.append(emotion_idx)
        else:
            x_test.append(np.array(img))
            y_test.append(emotion_idx)
        
        print("Extracting images data... {:.1f}%".format(100.*(i+1)/nb_imgs), end='\r')
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test
 

#### SIFT FEATURES ####

def detect_face_landmarks(gray_img, detector, predictor):
    """Detects 51 facial landmarks (eyes, eyebrows, nose, mouth) using dlib.using
    """
    # detect face in the grayscale image
    rects = detector(gray_img, 1)
    
    if len(rects)==0:
        # if no face was detected, we set the face rectangle to the entire image
        rect = dlib.rectangle(0,0,gray_img.shape[0],gray_img.shape[1])
    else:
        rect = rects[0]

    # determine the facial landmarks for the face region
    face_landmarks = face_utils.shape_to_np(predictor(gray_img, rect))
    face_landmarks = face_landmarks[17:] # remove landmarks around the face (useless)
    
    return face_landmarks

def compute_sift(img, detector, predictor):
    """Computes the SIFT descriptors of each of the facial landmarks of the given image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_landmarks = detect_face_landmarks(gray, detector, predictor)
    if face_landmarks is None:
        return None
    
    # Convert the detected face landmarks to KeyPoint to use with SIFT
    kp_conv = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=16) for pt in face_landmarks]

    sift = cv2.xfeatures2d.SIFT_create()
    sift_descriptors = sift.compute(gray, kp_conv)[1]

    return sift_descriptors

def extract_all_sift_features(x_data):
    """Extracts all the SIFT descriptors of the facial landmarks 
       of every frame in the given dataset.
    """
    # Initialize dlib's face detector and create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    #x_int = np.uint8(np.multiply(x_data, 256))
    x_int = np.uint8(x_data)
    sift_features = []
    
    for (i, img) in enumerate(x_int):
        sift = compute_sift(img, detector, predictor)
        if sift is None:
            print('\nWarning: No face detected!')
        sift_features.append(sift)
        
        print("Extracting SIFT features... {:.1f}%".format(100.*(i+1)/x_int.shape[0]), end='\r')
    
    print('')
    sift_features = np.array(sift_features)
    shp = sift_features.shape
    sift_features = sift_features.reshape((shp[0], -1))
    sift_features = np.multiply(sift_features, 1./255)
    
    return sift_features


#### VGG MODEL ####

def get_vgg_custom(vgg_weights_path, output_layer=None):
    vgg = load_model(custom_vgg_weights_path)
    if output_layer is not None:
        output = vgg.get_layer(output_layer).output
        vgg = Model(inputs=vgg.input, outputs=output)
        
    return vgg