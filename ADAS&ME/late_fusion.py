import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, LSTM, SimpleRNN, Bidirectional, BatchNormalization, GlobalAveragePooling2D, concatenate
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers

from training import train_leave_one_out, leave_one_out_split, DataGenerator, CustomVerbose, get_class_weight
from testing import evaluate_model, print_model_eval_metrics
from plot import plot_histories, plot_confusion_matrix
from const import *

def create_tcnn_top(pre_trained_model_path=None):
    """Create the top of the tcnn with fully connected layers.
    """
    def inner():
        input_shape=(7, 7, 512)

        tcnn_top = Sequential()
        tcnn_top.add(Convolution2D(1024, (7, 7), activation='relu', name='fc6', input_shape=input_shape))
        tcnn_top.add(Dropout(0.5))
        tcnn_top.add(Convolution2D(512, (1, 1), activation='relu', name='fc7b'))
        tcnn_top.add(Dropout(0.5))
        tcnn_top.add(Convolution2D(nb_emotions, (1, 1), name='fc8b'))
        tcnn_top.add(Flatten())
        tcnn_top.add(Activation('softmax'))

        if pre_trained_model_path is not None:
            tcnn_top.load_weights(pre_trained_model_path, by_name=True)
            tcnn_top.layers[0].trainable = False
        
        tcnn_top.compile(loss='categorical_crossentropy',
                     optimizer=optimizers.Adam(),
                     metrics=['accuracy'])
        
        return tcnn_top
    return inner

def create_phrnn_model(features_per_lm):
    def phrnn_creator():
        # Define inputs
        eyebrows_input = Input(shape=(5, 10*features_per_lm), name='eyebrows_input')
        nose_input = Input(shape=(5, 9*features_per_lm), name='nose_input')
        eyes_input = Input(shape=(5, 12*features_per_lm), name='eyes_input')
        out_lip_input = Input(shape=(5, 12*features_per_lm), name='out_lip_input')
        in_lip_input = Input(shape=(5, 8*features_per_lm), name='in_lip_input')

        # First level of BRNNs
        eyebrows = Bidirectional(SimpleRNN(40, return_sequences=True), name='BRNN40_1')(eyebrows_input)
        nose = Bidirectional(SimpleRNN(40, return_sequences=True), name='BRNN40_2')(nose_input)
        eyes = Bidirectional(SimpleRNN(40, return_sequences=True), name='BRNN40_3')(eyes_input)
        in_lip = Bidirectional(SimpleRNN(40, return_sequences=True), name='BRNN40_4')(in_lip_input)
        out_lip = Bidirectional(SimpleRNN(40, return_sequences=True), name='BRNN40_5')(out_lip_input)

        eyebrows_nose = concatenate([eyebrows, nose])
        eyes_in_lip = concatenate([eyes, in_lip])

        # Second level of BRNNs
        eyebrows_nose = Bidirectional(SimpleRNN(64, return_sequences=True), name='BRNN64_1')(eyebrows_nose)
        eyes_in_lip = Bidirectional(SimpleRNN(64, return_sequences=True), name='BRNN64_2')(eyes_in_lip)
        out_lip = Bidirectional(SimpleRNN(64, return_sequences=True), name='BRNN64_3')(out_lip)

        eyes_lips = concatenate([eyes_in_lip, out_lip])

        # Third level of BRNNs
        eyebrows_nose = Bidirectional(SimpleRNN(90, return_sequences=True), name='BRNN90_1')(eyebrows_nose)
        eyes_lips = Bidirectional(SimpleRNN(90, return_sequences=True), name='BRNN90_2')(eyes_lips)

        output = concatenate([eyebrows_nose, eyes_lips])

        # Final BLSTM and fully-connected layers
        output = Bidirectional(LSTM(80), name='BLSTM')(output)
        output = Dense(128, activation="relu", name='fc1')(output)
        output = Dense(nb_emotions, activation="softmax", name='fc2')(output)

        inputs = [eyebrows_input, nose_input, eyes_input, in_lip_input, out_lip_input]
        phrnn = Model(inputs=inputs, outputs=output)

        phrnn.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])
        return phrnn
    
    return phrnn_creator

###########################
######## MAIN CODE ########
###########################

#### TRAINING ####

files_per_batch=1
save_best_model = True

# Create the callbacks
#custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=6, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]

data_path = '/Volumes/External/ProjectData/ADAS&ME/data'
annotations_path = '/Volumes/External/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
files_list = FILES_LIST
class_weight = get_class_weight(files_list, annotations_path)
#class_weight = None


#Start training PHRNN
print('Training PHRNN model...')
epochs = 40
phrnn_model_path = 'models/late_fusion/phrnn/phrnn1.h5'
features = 'sift-phrnn'
phrnn, histories = train_leave_one_out(create_phrnn_model(features_per_lm=128),
                                       data_path,
                                       features,
                                       annotations_path,
                                       files_list, 
                                       files_per_batch, 
                                       epochs, 
                                       callbacks, 
                                       class_weight,
                                       save_best_model=save_best_model, 
                                       model_path=phrnn_model_path)

plot_histories(histories, 'PHRNN Model - ADAS&ME')


#Start training TCNN
print('Training TCNN model...')
epochs = 10
tcnn_model_path = 'models/late_fusion/tcnn/tcnn1.h5'
features = 'vgg-tcnn'
pre_trained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ck-fer/models/tcnn/tcnn_split1.h5'

tcnn_top, histories = train_leave_one_out(create_tcnn_top(pre_trained_model_path),
                                       data_path,
                                       features,
                                       annotations_path,
                                       files_list, 
                                       files_per_batch, 
                                       epochs, 
                                       callbacks, 
                                       class_weight,
                                       save_best_model=save_best_model, 
                                       model_path=tcnn_model_path)

plot_histories(histories, 'TCNN Model - ADAS&ME')


#### TESTING ####

print('\nTesting model...')

y_pred, y_true = evaluate_tcnn_phrnn_model(tcnn_model_path, phrnn_model_path, data_path, annotations_path, files_list, files_per_batch, merge_weight=0.5)
print_model_eval_metrics(y_pred, y_true)
