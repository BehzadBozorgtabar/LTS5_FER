import numpy as np

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

from keras.models import load_model

"""
Setups the jetson GPU to avoid memory error during the predictions
"""
def init_gpu_session(gpu_fraction=0.333):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
	ktf.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

"""
Loads the model used for prediction
Arguments:
	model_path	: The path of the model used for predictions
Returns:
	The loaded model
"""
def init_model(model_path):
	return load_model(model_path)


"""
Makes the prediction for the input image.
Arguments:
	- model 	: The loaded model used for prediction
	- image 	: The input image, it has to be a 1*224*224*3 numpy array. Plus the values have to be between 0 and 1 instead of 0 and 255
	- emotions 	: A dictionnary containing the possible predictions
Returns:
	- emotion 	: The emotion name predicted (String)
	- predicted_class : The corresponding index of the prediction (Int)
	- confidence	: The confidence of the prediction (String)
	- predictions	: The probabilities for each emotion
"""
def predict(model, image, emotions = {1: "Neutral", 2: "Positive", 3: "Frustrated", 4: "Anxiety"}):
	predictions = model.predict(image).reshape(-1)
	predicted_class = np.argmax(predictions) + 1
	emotion = emotions[predicted_class]
	confidence = "%.2f" % float(predictions[predicted_class-1])
	return emotion, int(predicted_class), confidence, predictions
