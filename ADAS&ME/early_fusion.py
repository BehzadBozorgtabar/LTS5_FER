import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, LSTM, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers

from training import train_leave_one_out, leave_one_out_split, DataGenerator, CustomVerbose, get_class_weight
from testing import evaluate_model, print_model_eval_metrics
from plot import plot_histories, plot_confusion_matrix
from const import *

def create_lstm_finetuning(pretrained_model_path, optimizer):
    def inner():
        lstm = load_model(pretrained_model_path)
        #lstm.layers[0].trainable=False # Do no train lstm layer
        
        output = Dense(nb_emotions, activation="softmax", name='dense_emo')(lstm.layers[-2].output)
        lstm_ft = Model(inputs=lstm.input, outputs=output)
            
        lstm_ft.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return lstm_ft
        
    return inner

def create_lstm(optimizer):
	"""Creates the LSTM model that goes on top of VGG and SIFT.
	"""
	def inner():
		lstm_units = 32
		hidden_units = 16
		input_dim = 7552  # 1024 for vgg + 6528 for sift
		nb_frames = 5

		input_shape = (nb_frames, input_dim)

		lstm = Sequential()
		lstm.add(LSTM(lstm_units, input_shape=input_shape))
		lstm.add(Dropout(0.5))
		lstm.add(Dense(hidden_units, activation="relu"))
		lstm.add(Dropout(0.5))
		lstm.add(Dense(nb_emotions, activation="softmax", name='dense_emo'))

		lstm.compile(loss='categorical_crossentropy',
					  optimizer=optimizer,
					  metrics=['accuracy'])
		return lstm
	return inner

###########################
######## MAIN CODE ########
###########################

#### TRAINING ####

epochs=10
files_per_batch=1
save_best_model = True
model_path = 'models/early_fusion/early_fusion1.h5'
#pretrained_model_path = 'models/vgg-sift-lstm2_split1.h5'
pretrained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/mug-fer/models/vgg-sift-lstm/vgg-sift-lstm2_split1.h5'
#pretrained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ck-fer/models/vgg-sift-lstm/vgg-sift-lstm_split1.h5'

# Create the callbacks
custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=6, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]

data_path = '/Volumes/External/ProjectData/ADAS&ME/data'
#data_path = 'data'
features = 'vgg-sift'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
files_list = FILES_LIST
class_weight = get_class_weight(files_list, annotations_path)
#class_weight = None

#Start training
#create_lstm_finetuning(pretrained_model_path, optimizers.Adam())
#create_lstm(optimizers.Adam())
lstm, histories = train_leave_one_out(create_lstm_finetuning(pretrained_model_path, optimizers.Adam()),
                                       data_path,
                                       features,
                                       annotations_path,
                                       files_list, 
                                       files_per_batch, 
                                       epochs, 
                                       callbacks, 
                                       class_weight,
                                       save_best_model=save_best_model, 
                                       model_path=model_path)

plot_histories(histories, 'Early Fusion Model - ADAS&ME')


#### TESTING ####

print('\nTesting model...')

y_pred, y_true = evaluate_model(model_path, data_path, annotations_path, files_list, files_per_batch)
print_model_eval_metrics(y_pred, y_true)
