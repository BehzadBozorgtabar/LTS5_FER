import numpy as np

img_size = 224 # resize images to 224*224 px

# emotions = ['Neutral', 'Positive','Frustration','Anxiety']
# nb_emotions = len(emotions)

emotions = ['Neutral', 'Positive','Negative']
nb_emotions = len(emotions)
emo_conv = {0:0, 1:1, 2:2, 3:2}

# Define the masks corresponding to each cluster of landmarks
lm_range = np.array(range(51))
eyebrows_mask = (lm_range < 10) # 10 landmarks
nose_mask = np.logical_and(10 <= lm_range, lm_range < 19) # 9 landmarks
eyes_mask = np.logical_and(19 <= lm_range, lm_range < 31)# 12 landmarks
outside_lip_mask = np.logical_and(31 <= lm_range, lm_range < 43) # 12 landmarks
inside_lip_mask = (43 <= lm_range) # 8 landmarks
nose_center_idx = 13

face_predictor_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/data/shape_predictor_68_face_landmarks.dat'
vggCustom_weights_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/models/customised_VGG.h5'
