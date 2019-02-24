nb_channels = 3
SMB_HEADER_SIZE = 20

custom_vgg_weights_path = 'models/customised_VGG.hdf5'
custom_resnet_weights_path = 'models/RAF_RAF_RESNET.hdf5'
custom_squeezeNet_weights_path = 'models/SqueezeNet.hdf5'

emotions = ['Neutral', 'Positive', 'Frustrated', 'Anxiety']
emotionsDic = {'Neutral' : 1, 'Positive' : 2, 'Frustrated' : 3, 'Anxiety' : 4}

#emotions = ['Neutral', 'Positive', 'Negative']
#emotionsDic = {'Neutral' : 1, 'Positive' : 2, 'Negative': 3}
nb_emotions = len(emotions)

models_path = 'models/'
vgg_model_path = 'VGG_model/'
resnet_model_path = 'ResNet_model/'
squeezeNet_model_path = 'SqueezeNet_model/'
results_path = 'results/'

csv_smb_files_path = "Annotations_to_pkl/"
images_files_path = "Images_to_pkl/"
real_data_path = "data/Real_data/"

MAX_SEGMENT_SIZE = 1000

"""
To modify: Fill the list with all subjects you don't want to test
"""
no_test_data = ['S000', 'S004', 'S005', 'S007', 'S008', 'S011', 'S015', 'S022', 'S026']
