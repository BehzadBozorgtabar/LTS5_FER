import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cross_validation import StratifiedKFold

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, LSTM, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.applications.densenet import DenseNet121

from extract import extract_frames, extract_data, extract_model_features
from training import create_vgg_fc7, train_crossval, CustomVerbose
from sift import extract_all_sift_features
from plot import plot_histories, plot_confusion_matrix
from const import *


def create_lstm_densenet_sift_model():
    """Creates the LSTM model that goes on top of DenseNet and SIFT.
    """
    lstm_units = 32
    hidden_units = 16
    input_dim = 7552  # 1024 for DenseNet + 6528 for sift

    input_shape = (nb_frames, input_dim)  # 4096 for vgg + 8704 for sift

    lstm = Sequential()
    lstm.add(LSTM(lstm_units, input_shape=input_shape))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(hidden_units, activation="relu"))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(nb_emotions, activation="softmax"))

    lstm.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.SGD(),
                 metrics=['accuracy'])
    return lstm


######## MAIN CODE ########

# If not already done, we extract the relevant frames from the raw MUG dataset
#extract_frames(subjects_path, frames_path)

# Now we extract the training and target data from the frames
x, y = extract_data(frames_path)

# Create DenseNet model
densenet = DenseNet121(include_top=False, input_shape=(img_size, img_size, nb_channels))
densenet = Model(inputs=densenet.input, output=GlobalAveragePooling2D(name='avg_pool')(densenet.output))

# Extract the DenseNet features
#densenet_features = extract_model_features(densenet, x)
with open('data/densenet_features.pkl', 'rb') as f:
    densenet_features = pickle.load(f)

# Extract SIFT features
#sift_features = extract_all_sift_features(x)
with open('data/sift_features.pkl', 'rb') as f:
    sift_features = pickle.load(f)

# Concatenate the DenseNet features with the SIFT features
densenet_sift_features = np.concatenate([densenet_features, sift_features], axis=2)


## TRAINING ##
batch_size=32
epochs=300
n_folds=5
save_best_model = True
weights_path = 'models/densenet-sift-lstm/model1.h5'

# Create the callbacks
custom_verbose = CustomVerbose(epochs)
early_stop = EarlyStopping(patience=50)
model_checkpoint = ModelCheckpoint(weights_path,
                                   monitor='val_acc',
                                   save_best_only=True,
                                   save_weights_only=True)
callbacks = [custom_verbose, early_stop, model_checkpoint] if save_best_model else [custom_verbose, early_stop]

lstm_densenet_sift, skf, histories = train_crossval(create_lstm_densenet_sift_model,
                                                    densenet_sift_features,
                                                    y,
                                                    batch_size=batch_size,
                                                    epochs=epochs,
                                                    callbacks=callbacks,
                                                    n_folds=n_folds,
                                                    save_best_model=save_best_model,
                                                    weights_path=weights_path)

plot_histories(histories, 'DenseNet-SIFT-LSTM, {}-fold cross-validation'.format(n_folds))


## LOADING PRE-TRAINED MODEL ##
# lstm = create_lstm_densenet_sift_model()
# lstm.load_weights('models/densenet-sift_lstm/model5-0.95acc.h5')