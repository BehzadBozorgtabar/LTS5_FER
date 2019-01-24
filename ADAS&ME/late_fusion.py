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

from training import create_tcnn_top, create_squeezenet_tcnn_top, create_phrnn_model, train_leave_one_out, leave_one_out_split, DataGenerator, CustomVerbose
from testing import evaluate_model, evaluate_tcnn_phrnn_model, print_model_eval_metrics
from plot import plot_histories, plot_confusion_matrix
from const import *


###########################
######## MAIN CODE ########
###########################

#### TRAINING ####

# insert path of the frames & annotations
frames_data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/Real_data5'

# insert path of pre-extracted features here
data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/data/data5'

# Subjects to be used for training
SUBJECTS = ['S000','S004','S007','S008','S011','S015','S022','S026', \
            'TS4_DRIVE','TS7_DRIVE','TS10_DRIVE', \
            'Vedecom1','Vedecom2','Vedecom3',\
            'VP03','VP17','VP18']

phrnn_features = None

subjects = SUBJECTS


#### PHRNN ####

save_best_model = True

# Start training PHRNN
print('\nTraining PHRNN model...')
epochs = 200

# Create the callbacks
#custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=100, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]

phrnn_model_path = 'models/late_fusion/phrnn/landmarks-phrnn2.h5'
phrnn_features = 'landmarks-phrnn' # 'landmarks-phrnn' or 'sift-phrnn'
phrnn_features_per_lm = 128 if 'sift' in phrnn_features else 2

phrnn, hist_phrnn = train_leave_one_out(create_phrnn_model(phrnn_features_per_lm),
                                       data_path,
                                       frames_data_path,
                                       subjects,
                                       phrnn_features, 
                                       epochs, 
                                       callbacks, 
                                       save_best_model=save_best_model, 
                                       model_path=phrnn_model_path)

# plot_histories(hist_phrnn, 'PHRNN Model - ADAS&ME')


#### TCNN ####

save_best_model = True

#Start training TCNN
print('\n\nTraining TCNN model...')
epochs = 50

# Create the callbacks
custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=40, monitor='val_loss')
callbacks = [early_stop, custom_verbose]

tcnn_model_path = 'models/late_fusion/tcnn/tcnn2.h5'
# tcnn_model_path = 'models/late_fusion/tcnn/squeezenet_tcnn1.h5'
tcnn_features = 'vgg-tcnn' #'vgg-tcnn' or 'squeezenet-tcnn'

tcnn_top, hist_tcnn = train_leave_one_out(create_squeezenet_tcnn_top,#create_tcnn_top() or create_squeezenet_tcnn_top
                                         data_path,
                                         frames_data_path,
                                         subjects,
                                         tcnn_features, 
                                         epochs, 
                                         callbacks, 
                                         save_best_model=save_best_model, 
                                         model_path=tcnn_model_path)


plot_histories(hist_phrnn, 'PHRNN Model - ADAS&ME')
plot_histories(hist_tcnn, 'TCNN Model - ADAS&ME')


#### TESTING ####

print('\nTesting model...')
merge_weight = 0.45

# set tcnn_model_path to None to test only PHRNN and vice-versa
y_pred, y_true = evaluate_tcnn_phrnn_model(tcnn_model_path, phrnn_model_path, tcnn_features, phrnn_features, subjects, data_path, frames_data_path, merge_weight=merge_weight)
print_model_eval_metrics(y_pred, y_true)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_emotions)))
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=True)
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=False)
