from online_stream import online_prediction
from offline_stream import offline_prediction

"""
Example of implementation 
"""
    

if __name__ == "__main__":

	arg = sys.argv[1]
	if arg in ['offline, online']:

		if arg == 'online':
			### ONLINE PREDICTION ###
			type_model = "vgg" #Type vgg or squeezenet
			model_name = "data/model_VP03.01-2.54.hdf5" #model name you want to use for the prediction

			input_host = "169.254.163.109" #ip address of the smarteye camera
			input_port = 5005 #the port used to make a tcp connection with the smarteye camera
			width = 640 #the width of the image you want to read
			height = 480 #the height of the image you want to read

			online_prediction(model_name, type_model, input_host, input_port, width, height, mqttHost="localhost", client_name="JSON", port=1883, topic='emotion')

		else:
			### OFFLINE PREDICTION ###
			test_file = "data/20180814_101543.smb" #The path of the SMB file
			type_model = "squeezenet" #Type vgg or squeezenet
			model_name = "data/model_VP03.01-2.54.hdf5" #model name you want to use for the prediction

			offline_prediction(model_name, type_model, test_file, mqttHost="test.mosquitto.org", client_name="JSON", port=1883, topic='emotion', SMB_HEADER_SIZE = 20)
