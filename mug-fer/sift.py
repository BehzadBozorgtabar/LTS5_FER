import numpy as np

from imutils import face_utils
import dlib
import cv2
 
def detect_face_landmarks(gray_img, detector, predictor):
    """Detects 51 facial landmarks (eyes, eyebrows, nose, mouth) using dlib.using
    """
    # detect face in the grayscale image
    rect = detector(gray_img, 1)[0]

    # determine the facial landmarks for the face region
    face_landmarks = face_utils.shape_to_np(predictor(gray_img, rect))
    face_landmarks = face_landmarks[17:] # remove landmarks around the face (useless)
    
    return face_landmarks

def compute_face_landmarks(x_data):
    """Computes all the facial landmarks of every frame in the given dataset.
    """
    # Initialize dlib's face detector and create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    x_int = np.uint8(np.multiply(x_data, 256))
    face_landmarks = []

    for (i, sample) in enumerate(x_int):
        lm_seq = []
        for frame in sample:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lm = detect_face_landmarks(gray, detector, predictor)
            
            lm_seq.append(lm)
        face_landmarks.append(np.array(lm_seq))
        print("Computing facial landmarks... {:.1f}%".format(100.*(i+1)/x_int.shape[0]), end='\r')
    
    print('')

    face_landmarks = np.array(face_landmarks)
    
    return face_landmarks

def normalize_face_landmarks(face_landmarks, nose_landmark_idx):
    """Normalize facial landmarks by subtracting the coordinates corresponding
       to the nose, and dividing by the standard deviation.
    """
    face_landmarks_norm = np.zeros(face_landmarks.shape)
    
    for (i, seq) in enumerate(face_landmarks):
        for (j, lm) in enumerate(seq):
            face_landmarks_norm[i,j] = lm - lm[nose_landmark_idx]
            
    std_x = np.std(face_landmarks_norm[:,:,:,0].reshape((-1,)))
    std_y = np.std(face_landmarks_norm[:,:,:,1].reshape((-1,)))
    
    face_landmarks_norm[:,:,:,0] = np.multiply(face_landmarks_norm[:,:,:,0], 1./std_x)
    face_landmarks_norm[:,:,:,1] = np.multiply(face_landmarks_norm[:,:,:,1], 1./std_y)
    
    return face_landmarks_norm

def compute_sift(img, detector, predictor):
    """Computes the SIFT descriptors of each of the facial landmarks of the given image.
    """
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
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    x_int = np.uint8(np.multiply(x_data, 256))
    sift_features = []

    for sample in x_int:
        sift_seq = []
        for frame in sample:
            sift = compute_sift(frame, detector, predictor)
            sift_seq.append(sift)
        sift_features.append(np.array(sift_seq))

    sift_features = np.array(sift_features)
    shp = sift_features.shape
    sift_features = sift_features.reshape((shp[0], shp[1], -1))
    sift_features = np.multiply(sift_features, 1./255)
    
    return sift_features
