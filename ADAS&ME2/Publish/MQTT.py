import paho.mqtt.client as mqtt
import json

"""
Callback function when connected to the MQTT
This function is automatically called during the connection step, you don't need to use it
"""
def on_connect(client, userdata, flags, rc):
	if rc == 0:
		client.connected_flag = True
		print("Connection successful")
	else:
		print("Connection unsuccessful, returned code = ",rc)
		sys.exit()


"""
Connects the client to the MQTT broker for the publication step
Arguments:
	mqttHost	: The host name or the ip address of the mqtt broker used for publication
	client_name	: The name of the client to make the connection with the mqtt broker
	port		: The port used for the tcp connection with the broker
Returns:
	The client connected to the MQTT broker
"""
def init_mqtt_connection(mqttHost, client_name, port):
	### CONNECTION TO MQTT BROKER ###
	mqtt.Client.connected_flag = False

	client = mqtt.Client(client_name)
	client.on_connect = on_connect
	client.loop_start()
	print("Connecting to broker ", mqttHost)
	client.connect(mqttHost, port = port)
	while not client.connected_flag:
		print("Waiting...")
		time.sleep(1)
	print("Waiting...")
	client.loop_stop()
	print("Starting main program")

	return client


"""
The client connected to the MQTT broker publishes the output on the topic "topic"
Arguments:
	- client	: The client connected to the MQTT broker
	- output	: The output of the algorithm
	- topic		: The topic on which we want to publish the output to the MQTT broker		
"""
def publish_output(client, output, topic):
	output = json.dumps(output)
	client.publish(topic, output)


"""
Disconnects the client connected to the MQTT broker
Argument:
	- client	: The client connected to the MQTT broker
"""
def disconnect_client(client):
	client.disconnect()
