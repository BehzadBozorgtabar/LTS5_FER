# LTS5_FER
Real-time face emotion detection using CNN models

Please follow the instructions below to use the scripts. The script you need to execute is main.py with the instruction: python main.py. Python 2 is used.

1) Add the models you want to use in the data folder

2) You only need to modify the script main.py. 

	-	line 6: specify if you want to use a vgg model typing "vgg" or a squeezeNet model typing "squeezenet"

	-	line 7: specify the name of the model you want to use for the prediction. Be careful that the model you're using is a custom model of one of the 2 CNNs you need to specify on line 6

	-	line 17: write the ip address of the smart eye camera to make a TCP connection with it

	-	line 18: write the destination port to make a TCP connection with the smart eye camera

	-	line 19: specify the width of the image you want to read

	-	line 20: specify the height of the image you want to read

	-	line 22: beside mqttHost, type the hostname or the ip address of the mqtt broker for the publication step. Also, specify a client name and the port number.

	-	line 23: specify the name of the topic on which you want to publish the predictions to the mqtt broker 
