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
from sift import extract_all_sift_features
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
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
files_list = [  ('TS4_DRIVE', '20180824_150225'),
                 ('TS4_DRIVE', '20180824_150302'),
                 ('TS4_DRIVE', '20180824_150401'),
                 ('TS4_DRIVE', '20180824_150443'),
                 ('TS4_DRIVE', '20180824_150543'),
                 ('TS4_DRIVE', '20180829_135202'),
                ('TS6_DRIVE', '20180828_162845'),
                 ('TS6_DRIVE', '20180828_163043'),
                 ('TS6_DRIVE', '20180828_163156'),
                 ('TS6_DRIVE', '20180829_083936'),
                 ('TS6_DRIVE', '20180829_084019'),
                 ('TS6_DRIVE', '20180829_084054'),
                 ('TS6_DRIVE', '20180829_084134'),
                 ('TS6_DRIVE', '20180829_091659'),
                ('TS7_DRIVE', '20180828_142136'),
                 ('TS7_DRIVE', '20180828_142244'),
                 ('TS7_DRIVE', '20180828_142331'),
                 ('TS7_DRIVE', '20180828_150143'),
                 ('TS7_DRIVE', '20180828_150234'),
                 ('TS7_DRIVE', '20180828_150322'),
                 ('TS7_DRIVE', '20180828_150425'),
                ('TS8_DRIVE',  '20180827_174358'),
                ('TS8_DRIVE',  '20180827_174508'),
                ('TS8_DRIVE',  '20180827_174552'),
                ('TS8_DRIVE',  '20180827_174649'),
                ('TS8_DRIVE',  '20180827_174811'),
                ('TS9_DRIVE',  '20180827_165431'),
                ('TS9_DRIVE',  '20180827_165525'),
                ('TS9_DRIVE',  '20180827_165631'),
                ('TS10_DRIVE', '20180827_164836'),
                ('TS10_DRIVE', '20180827_164916'),
                ('TS10_DRIVE', '20180827_165008'),
                ('TS10_DRIVE', '20180828_082231'),
                ('TS10_DRIVE', '20180828_082326'),
                ('TS10_DRIVE', '20180828_085245'),
                ('TS10_DRIVE', '20180828_114716'),
                ('TS10_DRIVE', '20180828_114905'),
                ('TS10_DRIVE', '20180828_143226'),
                ('TS10_DRIVE', '20180828_143304'),
                ('TS11_DRIVE', '20180827_115441'),
                ('TS11_DRIVE', '20180827_115509'),
                ('TS11_DRIVE', '20180827_115620'),
                ('TS11_DRIVE', '20180827_115730'),
                ('TS11_DRIVE', '20180827_115840'),
                ('TS11_DRIVE', '20180827_120035')]
class_weight = get_class_weight(files_list, annotations_path)
#class_weight = None

#Start training
#create_lstm_finetuning(pretrained_model_path, optimizers.Adam())
#create_lstm(optimizers.Adam())
lstm, histories = train_leave_one_out(create_lstm_finetuning(pretrained_model_path, optimizers.Adam()),
                                       data_path,
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
