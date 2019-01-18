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

extraction_path_to_image = "data_to_extract/to_images/"
extraction_path_to_pkl = "data_to_extract/to_pkl/"
real_data_path = "data/Real_data/"
cleanData_path = "DATASET/ValidImages/"
cleanImages_path = "DATASET/ImagesPickled/"
image_path = "DATASET/Images_to_treat/"

MAX_SEGMENT_SIZE = 1000
no_test_data = ['S000', 'S004', 'S005', 'S007', 'S008', 'S011', 'S015', 'S022', 'S026']
