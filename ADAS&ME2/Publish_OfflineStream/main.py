from test import init, prediction


if __name__ == '__main__':
	
	test_file = "data/20180814_101543.smb"
	model_name = "data/model_VP03.01-2.54.hdf5"

	model, client = init(model_name, mqttHost = "test.mosquitto.org", client_name= "JSON", port=1883)
	prediction(model, test_file, client, img_size=227, type_algorithm="SqueezeNet", topic='emotion') 
