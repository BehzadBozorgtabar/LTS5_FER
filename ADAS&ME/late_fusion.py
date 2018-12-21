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

from training import create_tcnn_top, create_phrnn_model, train_leave_one_out, leave_one_out_split, DataGenerator, CustomVerbose, get_class_weight
from testing import evaluate_model, evaluate_tcnn_phrnn_model, print_model_eval_metrics
from plot import plot_histories, plot_confusion_matrix
from const import *


###########################
######## MAIN CODE ########
###########################

#### TRAINING ####


frames_data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/Real_data3'
data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/data'

SUBJECTS = ['TS7_DRIVE', 'TS9_DRIVE', 'UC_B1', 'UC_B2', 'UC_B3', 'UC_B4', 'UC_B5','VP03', 'VP17', 'VP18']
CLASS_WEIGHT = {0: 0.3135634184068059,
                 1: 2.1480132450331126,
                 2: 10.199685534591195,
                 3: 4.04426433915212}

phrnn_features = None

subjects = SUBJECTS

class_weight = get_class_weight(subjects, frames_data_path, power=1)
print(class_weight)
#class_weight = CLASS_WEIGHT
#class_weight = None

#### PHRNN ####

save_best_model = True

# # Start training PHRNN
print('Training PHRNN model...')
epochs = 25

# Create the callbacks
#custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=12, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]

phrnn_model_path = 'models/late_fusion/phrnn/sift-phrnn1.h5'
phrnn_features = 'sift-phrnn' # 'landmarks-phrnn' or 'sift-phrnn'
phrnn_features_per_lm = 128 if 'sift' in phrnn_features else 2

phrnn, hist_phrnn = train_leave_one_out(create_phrnn_model(phrnn_features_per_lm),
                                       data_path,
                                       frames_data_path,
                                       subjects,
                                       phrnn_features, 
                                       epochs, 
                                       callbacks, 
                                       class_weight=class_weight,
                                       save_best_model=save_best_model, 
                                       model_path=phrnn_model_path)

# plot_histories(hist_phrnn, 'PHRNN Model - ADAS&ME')

#### TCNN ####

save_best_model = True

#Start training TCNN
print('Training TCNN model...')
epochs = 50

# Create the callbacks
custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=20, monitor='val_loss')
callbacks = [early_stop, custom_verbose]

tcnn_model_path = 'models/late_fusion/tcnn/tcnn1.h5'
tcnn_features = 'vgg-tcnn'
pre_trained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ck-fer/models/tcnn/tcnn_split1.h5'

tcnn_top, hist_tcnn = train_leave_one_out(create_tcnn_top(pre_trained_model_path),
                                         data_path,
                                         frames_data_path,
                                         subjects,
                                         tcnn_features, 
                                         epochs, 
                                         callbacks, 
                                         class_weight=class_weight,
                                         save_best_model=save_best_model, 
                                         model_path=tcnn_model_path)


plot_histories(hist_phrnn, 'PHRNN Model - ADAS&ME')
plot_histories(hist_tcnn, 'TCNN Model - ADAS&ME')


#### TESTING ####

print('\nTesting model...')
# tcnn_model_path, phrnn_model_path 
y_pred, y_true = evaluate_tcnn_phrnn_model(tcnn_model_path, phrnn_model_path, phrnn_features, subjects, data_path, frames_data_path)
print_model_eval_metrics(y_pred, y_true)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_emotions)))
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=True)
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=False)
