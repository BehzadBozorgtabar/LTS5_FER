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

from training import train_leave_one_out, leave_one_out_split, DataGenerator, CustomVerbose
from sift import extract_all_sift_features
from plot import plot_histories, plot_confusion_matrix
from const import *

def create_lstm_finetuning(pretrained_model_path, optimizer):
    def inner():
        lstm = load_model(pretrained_model_path)
        lstm.layers[0].trainable=False # Do no train lstm layer
        
        output = Dense(nb_emotions, activation="softmax", name='dense_emo')(lstm.layers[-2].output)
        lstm_ft = Model(inputs=lstm.input, outputs=output)
            
        lstm_ft.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return lstm_ft
        
    return inner

###########################
######## MAIN CODE ########
###########################

epochs=75
files_per_batch=1
save_best_model = True
model_path = 'models/early_fusion/early_fusion1.h5'
pretrained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/LTS5_FER/mug-fer/models/vgg-sift-lstm/vgg-sift-lstm2_split1.h5'

#data_path = '/Volumes/External/ProjectData/ADAS&ME/data'
data_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ADAS&ME/data'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
files_list = [  ('TS8_DRIVE',  '20180827_174358'),
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

# Create the callbacks
custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=20, monitor='val_loss')
callbacks = [custom_verbose, early_stop]

lstm, histories = train_leave_one_out(create_lstm_finetuning(pretrained_model_path, optimizers.Adam()),
                                       data_path,
                                       annotations_path,
                                       files_list, 
                                       files_per_batch, 
                                       epochs, 
                                       callbacks, 
                                       save_best_model=save_best_model, 
                                       model_path=model_path)

p = plot_histories(histories, 'VGG-SIFT-LSTM, ADAS&ME')