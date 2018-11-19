# Path of the raw data from the MUG dataset
subjects_path = '/Volumes/External/ProjectData/subjects3_cleaned'

# Path of the folder in which the selected frames should be stored
frames_path = 'MUG-data/frames'

vgg_weights_path = 'models/vgg_face_weights.h5'
vggCustom_weights_path = 'models/customised_VGG.h5'
vgg_features_path = 'data/vgg_features.pkl'
vgg_features_tcnn_path = 'data/vgg_custom_features_tcnn.pkl'
densenet_features_path = 'data/densenet_features.pkl'
sift_features_path = 'data/sift_features.pkl'
face_landmarks_path = 'data/face_landmarks.pkl'
y_data_path = 'data/y_data.pkl'

vgg_lstm_model_path = 'models/vgg-lstm.h5'
vgg_sift_lstm_model_path = 'models/vgg-sift-lstm/vgg-sift-lstm.h5'
densenet_lstm_model_path = 'models/densenet-lstm.h5'
densenet_sift_lstm_model_path = 'models/densenet-sift-lstm.h5'
tcnn_model_path = 'models/tcnn/tcnn.h5'
phrnn_model_path = 'models/phrnn/phrnn.h5'
sift_phrnn_model_path = 'models/phrnn/sift-phrnn.h5'

img_size = 224 # resize images to 224*224 px
nb_frames = 5 # Number of frames to select in each video
nb_channels = 3 # RGB images

emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
nb_emotions = len(emotions)