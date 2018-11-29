import numpy as np

from imutils import face_utils
import dlib
import cv2

face_predictor_path = '/Volumes/External/ProjectData/data/shape_predictor_68_face_landmarks.dat'

 
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
    img=img.astype(np.uint8)
    if len(img.shape)==2:
        gray=img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_landmarks = detect_face_landmarks(gray, detector, predictor)
    
    # Convert the detected face landmarks to KeyPoint to use with SIFT
    kp_conv = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=8) for pt in face_landmarks]
    
    sift = cv2.xfeatures2d.SIFT_create()
    sift_descriptors = sift.compute(gray, kp_conv)[1]
    
    return sift_descriptors

def extract_all_sift_features(x_data):
    """Extracts all the SIFT descriptors of the facial landmarks 
       of every frame in the given dataset.
    """
    # Initialize dlib's face detector and create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    
    x_int = np.uint8(np.multiply(x_data, 256))
    sift_features = []
    
    nb_f = len(x_data)
    for i, frame in enumerate(x_data):
        print(' Frame {}/{}'.format(i+1, nb_f), end='\r')
        sift = compute_sift(frame, detector, predictor)
        sift_features.append(sift)
            
    sift_features = np.array(sift_features)
    shp = sift_features.shape
    sift_features = sift_features.reshape((shp[0], shp[1], -1))
    sift_features = np.multiply(sift_features, 1./255)
    print('')
    
    return sift_features