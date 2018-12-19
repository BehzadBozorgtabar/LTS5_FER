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

data_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/data'
annotations_path = '/Volumes/Ternalex/ProjectData/ADAS&ME/ADAS&ME_data/annotated'
files_list = FILES_LIST

#### PHRNN ####

files_per_batch=1
save_best_model = True

# # Create the callbacks
#custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=8, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]
class_weight = get_class_weight(files_list, annotations_path)


# # Start training PHRNN
# print('Training PHRNN model...')
# epochs = 12
# phrnn_model_path = 'models/late_fusion/phrnn/sift-phrnn1.h5'
# features = 'sift-phrnn'
# phrnn, histories1 = train_leave_one_out(create_phrnn_model(features_per_lm=128),
#                                        data_path,
#                                        features,
#                                        annotations_path,
#                                        files_list, 
#                                        files_per_batch, 
#                                        epochs, 
#                                        callbacks, 
#                                        class_weight,
#                                        save_best_model=save_best_model, 
#                                        model_path=phrnn_model_path)
# plot_histories(histories1, 'PHRNN Model - ADAS&ME')


#### TCNN ####

files_per_batch=1
save_best_model = True

# Create the callbacks
#custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=80, monitor='val_loss')
callbacks = [early_stop]#, custom_verbose]
class_weight = get_class_weight(files_list, annotations_path, power=0.9)

#Start training TCNN
print('Training TCNN model...')
epochs = 400
tcnn_model_path = 'models/late_fusion/tcnn/tcnn1.h5'
features = 'vgg-tcnn'
pre_trained_model_path = '/Users/tgyal/Documents/EPFL/MA3/Project/fer-project/ck-fer/models/tcnn/tcnn_split1.h5'

# tcnn_top, histories2 = train_leave_one_out(create_tcnn_top(pre_trained_model_path),
#                                        data_path,
#                                        features,
#                                        annotations_path,
#                                        files_list, 
#                                        files_per_batch, 
#                                        epochs, 
#                                        callbacks, 
#                                        class_weight,
#                                        save_best_model=save_best_model, 
#                                        model_path=tcnn_model_path)


# plot_histories(histories1, 'PHRNN Model - ADAS&ME')
# plot_histories(histories2, 'TCNN Model - ADAS&ME')


#### TESTING ####

print('\nTesting model...')
files_list = [('Driver2', '20181121_162205'),
                 ('Driver3', '20180823_095005'),
                 ('TS7_DRIVE', '20180829_114712'),
                 ('UC_B', '20181121_163847'),
                 ('UC_B', '20181121_165322'),
                 ('VP03', '20180822_162114'),
                 ('VP17', '20180823_095005'),
                 ('VP18', '20180823_095958')]
# tcnn_model_path, phrnn_model_path
# y_pred, y_true = evaluate_tcnn_phrnn_model(tcnn_model_path, None, data_path, annotations_path, files_list, files_per_batch, merge_weight=0.5)
print_model_eval_metrics(y_pred, y_true)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(nb_emotions)))
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=True)
plot_confusion_matrix(cm, emotions, title='Late Fusion Model  -  ADAS&ME', normalize=False)
