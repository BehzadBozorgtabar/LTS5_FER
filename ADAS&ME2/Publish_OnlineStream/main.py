from test import init, prediction
    

if __name__ == "__main__":

	type_model = "vgg" #Type vgg or squeezenet
	model_name = "data/model_VP03.01-2.54.hdf5" #model name you want to use for the prediction

	if type_model == "vgg":
		type_algorithm = "Custom_VGG_FACE"
		img_size = 224
	elif type_model == "squeezenet":
		type_algorithm = "Custom_Squeeze_Net"
		img_size = 227


	input_host = "169.254.163.109" #ip address of the smarteye camera
	input_port = 5005 #the port used to make a tcp connection with the smarteye camera
	width = 640 #the width of the image you want to read
	height = 480 #the height of the image you want to read

	model, output_client = init(model_name, mqttHost = "test.mosquitto.org", client_name= "JSON", port=1883)
	prediction(model, output_client, input_host, input_port, width, height, img_size=img_size, type_algorithm=type_algorithm, topic='emotion') 

