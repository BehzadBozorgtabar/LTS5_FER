from test import init, prediction
from extraction import create_squeezenet_tcnn_bottom, create_tcnn_bottom, get_conv1_weights, get_conv_1_1_weights
from const import *

from keras.models import load_model


if __name__ == '__main__':

	tcnn_type = "vgg" # 'vgg' or 'squeezenet'
	
	test_file = "/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ADAS&ME/ADAS&ME_data/smb/TS?_DRIVE/20180814_101543.smb"

	# sift_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/sift-phrnn1_TS4_DRIVE.h5'
	lm_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/landmarks-phrnn1_TS4_DRIVE.h5'
	vgg_tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/tcnn1_TS4_DRIVE.h5'
	squeezenet_tcnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/tcnn/squeezenet_tcnn1_TS4_DRIVE.h5'

	phrnn_model = load_model(lm_phrnn_model_path)

	if tcnn_type == 'vgg':
		tcnn_model = load_model(vgg_tcnn_model_path)

		# Prepare TCNN extractor
		conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
		tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)
		tcnn_extractor = tcnn_bottom.predict

	elif tcnn_type == 'squeezenet':
		tcnn_model = load_model(squeezenet_tcnn_model_path)

		# Prepare TCNN extractor
		conv1_weights = get_conv1_weights(vggCustom_weights_path)
		tcnn_bottom = create_squeezenet_tcnn_bottom(squeezeNetCustom_weights_path, conv1_weights)
		tcnn_extractor = tcnn_bottom.predict


	client = init(mqttHost = "test.mosquitto.org", client_name= "JSON", port=1883)
	prediction(tcnn_model, phrnn_model, tcnn_extractor, test_file, client, img_size=224, tcnn_type=tcnn_type, phrnn_type="landmarks") 
