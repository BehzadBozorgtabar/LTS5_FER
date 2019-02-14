from test import init, prediction
from extraction import create_squeezenet_tcnn_bottom, create_tcnn_bottom, get_conv1_weights, get_conv_1_1_weights
from const import *

from keras.models import load_model
import keras.backend as K
import tensorflow as tf


if __name__ == '__main__':

	tcnn_type = "squeezenet" # 'vgg' or 'squeezenet'
	
	test_file = "data/20180814_101543.smb"

	#sift_phrnn_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/ADAS&ME/models/late_fusion/phrnn/sift-phrnn1_TS4_DRIVE.h5'
	lm_phrnn_model_path = 'data/landmarks-phrnn2_Vedecom1.h5'
	vgg_tcnn_model_path = 'data/tcnn2_Vedecom1.h5'
	squeezenet_tcnn_model_path = 'data/squeezenet_tcnn1_Vedecom1.h5'

	phrnn_model = load_model(lm_phrnn_model_path)

	if tcnn_type == 'vgg':
		tcnn_model = load_model(vgg_tcnn_model_path)

		# Prepare TCNN extractor
		print("Prepare TCNN extractor")
		conv1_1_weigths = get_conv_1_1_weights(vggCustom_weights_path)
		tcnn_bottom = create_tcnn_bottom(vggCustom_weights_path, conv1_1_weigths)
		tcnn_extractor = tcnn_bottom.predict
		img_size = 224

	elif tcnn_type == 'squeezenet':
		tcnn_model = load_model(squeezenet_tcnn_model_path)

		# Prepare TCNN extractor
		print("Prepare TCNN extractor")
		conv1_weights = get_conv1_weights(squeezeNetCustom_weights_path)
		tcnn_bottom = create_squeezenet_tcnn_bottom(squeezeNetCustom_weights_path, conv1_weights)
		tcnn_extractor = tcnn_bottom.predict
		img_size = 227

	input_host = "169.254.163.109"
	input_port = 5005
	width = 640
	height = 480

	output_client = init(mqttHost = "test.mosquitto.org", client_name= "JSON", port=1883)
	K.get_session().run(tf.global_variables_initializer())
	prediction(tcnn_model, phrnn_model, tcnn_extractor, output_client, input_host, input_port, width, height, img_size=img_size, tcnn_type=tcnn_type, phrnn_type="landmarks") 
