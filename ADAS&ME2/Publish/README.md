# LTS5_FER
Real-time face emotion detection using CNN models

Please follow the instructions below to use the scripts. The script you need to execute is main.py with the instruction: python main.py. Python 2 is used.

1) Add the models you want to use in the data folder, add the SMB test files in this folder if you want to use them.

2 Option1) The full pipeline for both online and offline prediction are already implemnted. You just need to modify the script main.py with 'offline' or 'online' as argument following main.py: 

	-	Online Prediction:
		-	line 9: Specify the type of the model ("either "vgg" or "squeezenet"")
		-	line 10: Specify the path of the model
		-	line 12: Specify the IP address of the smart eye camera
		-	line 13: Specify the port used to make a tcp connection with the smarteye camera
		-	line 14: Specify the width of the image you want to read
		-	line 15: Specify the height of the image you want to read
		-	line 17: Call the online prediction method and specify the IP address or the host name of the MQTT broker, 
		the client name, the port used for the connection and the topic on which we want to publish.

	-	Offline Prediction
		-	line 20: Specify the path of the SMB file
		-	line 21: Specify the type of the model ("either "vgg" or "squeezenet"")
		-	line 22: Specify the path of the model
		-	line 24: Call the online prediction method and specify the IP address or the host name of the MQTT broker, 
		the client name, the port used for the connection and the topic on which we want to publish. And also the SMB HEADER SIZE.

2 Option2) You want to implement your pipeline using the methods present in the scripts. See Part3 for full pipeline description and the comented
code in the test.py file.
