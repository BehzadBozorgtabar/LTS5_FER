# Path of the raw data from the MUG dataset
subjects_path = '/Volumes/External/ProjectData/subjects3_cleaned'

# Path of the folder in which the selected frames should be stored
frames_path = 'MUG-data/frames'

vgg_weights_path = 'models/vgg_face_weights.h5'
vgg_features_path = 'data/vgg_features.pkl'
densenet_features_path = 'data/densenet_features.pkl'
sift_features_path = 'data/sift_features.pkl'
y_data_path = 'data/y_data.pkl'

vgg_lstm_model_path = 'models/vgg-lstm.h5'
vgg_sift_lstm_model_path = 'models/vgg-sift-lstm.h5'
densenet_lstm_model_path = 'models/densenet-lstm.h5'
densenet_sift_lstm_model_path = 'models/densenet-sift-lstm.h5'

img_size = 224 # resize images to 224*224 px
nb_frames = 5 # Number of frames to select in each video
nb_channels = 3 # RGB images

emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
nb_emotions = len(emotions)