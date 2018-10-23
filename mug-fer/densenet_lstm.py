import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, LSTM, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.applications.densenet import DenseNet121

from extract import extract_frames, extract_data, extract_model_features
from training import create_vgg_fc7, train_crossval, CustomVerbose
from plot import plot_histories, plot_confusion_matrix
from const import *


def create_lstm_densenet_model():
    """Creates the LSTM model that goes on top of DenseNet.
    """
    lstm_units = 32
    hidden_units = 16
    input_dim = 1024

    input_shape = (nb_frames, input_dim)

    lstm = Sequential()
    lstm.add(LSTM(lstm_units, input_shape=input_shape))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(hidden_units, activation="relu"))
    lstm.add(Dropout(0.5))
    lstm.add(Dense(nb_emotions, activation="softmax"))

    lstm.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.Adam(),
                 metrics=['accuracy'])
    return lstm

###########################
######## MAIN CODE ########
###########################

# Train the model only if the call was made with argument 'train',
# otherwise, just test it
train = len(sys.argv) > 1 and sys.argv[1]=='train'

# If the densenet features were pre-computed, we don't have to recompute them
load_densenet_features = os.path.isfile(densenet_features_path)
load_frames = os.path.isdir(frames_path)


## EXTRACTION ##

# If not already done, we extract the relevant frames from the raw MUG dataset
if not load_frames:
    extract_frames(subjects_path, frames_path)


# Now we extract the training and target data from the frames
if not load_densenet_features:
    x, y = extract_data(frames_path)
else:
    with open(y_data_path, 'rb') as f:
        y = pickle.load(f)


if load_densenet_features:
    with open(densenet_features_path, 'rb') as f:
        densenet_features = pickle.load(f)
else:
    # Create DenseNet model
    densenet = DenseNet121(include_top=False, input_shape=(img_size, img_size, nb_channels))
    densenet = Model(inputs=densenet.input, output=GlobalAveragePooling2D(name='avg_pool')(densenet.output))

    # Extract the DenseNet features
    densenet_features = extract_model_features(densenet, x)


## TRAINING ##

if train:
  print("Training LSTM model...")
  batch_size=32
  epochs=500
  n_folds=5
  save_best_model = False
  weights_path='models/densenet-lstm2.h5'

  # Create the callbacks
  custom_verbose = CustomVerbose(epochs)
  early_stop = EarlyStopping(patience=50)
  model_checkpoint = ModelCheckpoint(weights_path,
                                     monitor='val_acc',
                                     save_best_only=True,
                                     save_weights_only=True)
  callbacks = [custom_verbose, early_stop, model_checkpoint] if save_best_model else [custom_verbose, early_stop]

  lstm_densenet, skf, histories = train_crossval(create_lstm_densenet_model,
                                                 densenet_features,
                                                 y,
                                                 batch_size=batch_size,
                                                 epochs=epochs,
                                                 callbacks=callbacks,
                                                 n_folds=n_folds,
                                                 save_best_model=save_best_model,
                                                 weights_path=weights_path)

  plot_histories(histories, 'DenseNet-LSTM, {}-fold cross-validation'.format(n_folds))

else:
  lstm_densenet = load_model(densenet_lstm_model_path)


## TESTING ##

# Get back the train/test split used
train_idx, test_idx = [],[]
labels = np.argmax(y, axis=1)
skf = StratifiedKFold(labels, n_folds=5, shuffle=False)
for i, (train, test) in enumerate(skf):
    train_idx.append(train)
    test_idx.append(test)

# Get emotion predictions
test_indices = test_idx[1]
y_predict = lstm_densenet.predict_classes(densenet_features[test_indices])
y_true = np.argmax(y[test_indices], axis=1)

# Computes the accuracy
acc = (y_predict==y_true).sum()/len(y_predict)
print('Test accuracy : {:.4f}'.format(acc))

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y[test_indices], axis=1), y_predict)
plot_confusion_matrix(cm, emotions, title='DenseNet-LSTM', normalize=True)

