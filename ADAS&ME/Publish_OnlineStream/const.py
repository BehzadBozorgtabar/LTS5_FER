import numpy as np

# Some constant values
nb_channels = 3
emotions = {1: "Neutral", 2: "Positive", 3: "Negative"}

face_predictor_path = 'data/shape_predictor_68_face_landmarks.dat'
vggCustom_weights_path = 'data/customised_VGG.h5'
squeezeNetCustom_weights_path = 'data/SqueezeNet.49-0.83.hdf5'


# Define the masks corresponding to each cluster of landmarks
lm_range = np.array(range(51))
eyebrows_mask = (lm_range < 10) # 10 landmarks
nose_mask = np.logical_and(10 <= lm_range, lm_range < 19) # 9 landmarks
eyes_mask = np.logical_and(19 <= lm_range, lm_range < 31)# 12 landmarks
outside_lip_mask = np.logical_and(31 <= lm_range, lm_range < 43) # 12 landmarks
inside_lip_mask = (43 <= lm_range) # 8 landmarks
nose_center_idx = 13
