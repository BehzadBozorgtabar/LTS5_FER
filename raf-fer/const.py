img_size = 224
nb_channels = 3
images_path = 'RAF-data/Image/aligned'
label_data_path = 'data/label_data.pkl'
label_path = 'RAF-data/EmoLabel/list_patition_label.txt'
emotions = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']

custom_vgg_weights_path = 'models/customised_VGG.hdf5'
sift_features_path = 'data/sift_features.pkl'
vgg_fc6_path = 'data/vgg_fc6_features.pkl'
vgg_fc7_path = 'data/vgg_fc7_features.pkl'

vgg_sift_svm_model_path = 'models/vgg_sift_svm.pkl'
sift_svm_model_path = 'models/sift_svm.pkl'
vgg_svm_model_path = 'models/vggfc6_svm.pkl'