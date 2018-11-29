import numpy as np
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, SimpleRNN, Bidirectional, LSTM, BatchNormalization, GlobalAveragePooling2D, average, concatenate
from keras.callbacks import Callback, ModelCheckpoint
from keras import optimizers

from const import *


def create_vgg_fc7(weights_path):
    """Create a VGG model with vgg-face weights and fc7 output.
    """
    input_shape=(img_size, img_size, nb_channels)

    vgg16 = Sequential()
    vgg16.add(ZeroPadding2D((1,1),input_shape=input_shape))
    vgg16.add(Convolution2D(64, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(64, (3, 3), activation='relu'))
    vgg16.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(128, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(128, (3, 3), activation='relu'))
    vgg16.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg16.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(ZeroPadding2D((1,1)))
    vgg16.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg16.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg16.add(Convolution2D(4096, (7, 7), activation='relu'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Convolution2D(4096, (1, 1), activation='relu', name='fc7'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Convolution2D(2622, (1, 1)))
    vgg16.add(Flatten())
    vgg16.add(Activation('softmax'))

    # Load the pre-trained VGG-face weights
    vgg16.load_weights(weights_path)

    for layer in vgg16.layers:
        layer.trainable = False

    # We want to take the ouput of the fc7 layer as input features of our next network
    fc7_output = vgg16.get_layer('fc7').output
    vgg16_fc7 = Model(inputs=vgg16.input, outputs=Flatten()(fc7_output))
    
    return vgg16_fc7


def train_crossval(create_model, x, y, batch_size, epochs, callbacks, n_folds, save_best_model=False, save_each_split=True, model_path=None):
    """Train a model using n-folds crossvalidation.
    """
    # Prepare for cross-validation
    labels = np.argmax(y, axis=1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)

    if save_best_model:
        cur_callbacks = callbacks.copy()
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True)
        cur_callbacks.append(model_checkpoint)

    histories = []
    val_accs = []
    for i, (train, test) in enumerate(skf.split(x, labels)):
        print("Running Fold {}/{}".format(i+1, n_folds))
        model = create_model()

        if save_best_model and save_each_split:
            cur_model_path = model_path[:-3]+'_split'+str(i+1)+model_path[-3:]
            model_checkpoint = ModelCheckpoint(cur_model_path, monitor='val_acc', save_best_only=True)
            cur_callbacks = callbacks.copy()
            cur_callbacks.append(model_checkpoint)

        hist = model.fit(x[train],
                         y[train],
                         validation_data = (x[test], y[test]),
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         callbacks=cur_callbacks if save_best_model else callbacks,
                         verbose=0)
        histories.append(hist)

        # Save the val_acc of the best epoch
        best_epoch = np.argmax(hist.history['val_acc'])
        best_val_acc = hist.history['val_acc'][best_epoch]
        val_accs.append(best_val_acc)
        print("  Best model : epoch {} - val_acc: {:.6f}".format(best_epoch, best_val_acc))

    cross_val_acc = np.mean(val_accs)
    print('Average validation accuracy : {:.6f}'.format(cross_val_acc))

    return model, skf, histories


class CustomVerbose(Callback):
    """Callback that prints a short 1-line verbose for each epoch and overwrites over the past one.
    """
    def __init__(self, epochs):
        self.epochs = epochs
        
    def on_train_end(self, logs={}):
        print('\n')
        
    def on_epoch_end(self, epoch, dictionary):        
        print('  Epoch %d/%d'%(epoch+1, self.epochs), ''.join([' - %s: %.6f'%tup for tup in dictionary.items()]), end="\r")


#### TCNN-PHRNN ####

def create_tcnn_bottom(vgg_weigths_path, conv1_1_weigths):
    """Creates the bottom part of the VGG-16 TCNN.
    """
    # Create inputs for the 5 frames
    input_shape=(img_size, img_size, nb_channels)
    frame1 = Input(shape=input_shape)
    frame2 = Input(shape=input_shape)
    frame3 = Input(shape=input_shape)
    frame4 = Input(shape=input_shape)
    frame5 = Input(shape=input_shape)
    
    # Convolution for each frame
    frame1_conv = ZeroPadding2D((1,1))(frame1)
    frame1_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1a')(frame1_conv)

    frame2_conv = ZeroPadding2D((1,1))(frame2)
    frame2_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1b')(frame2_conv)

    frame3_conv = ZeroPadding2D((1,1))(frame3)
    frame3_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1c')(frame3_conv)

    frame4_conv = ZeroPadding2D((1,1))(frame4)
    frame4_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1d')(frame4_conv)

    frame5_conv = ZeroPadding2D((1,1))(frame5)
    frame5_conv = Convolution2D(64, (3, 3), activation='relu', name='conv1_1e')(frame5_conv)
    
    # Temporal aggregation by averaging
    temp_aggr = average([frame1_conv, frame2_conv, frame3_conv, frame4_conv, frame5_conv])

    # Then standard VGG-16 architecture
    output = ZeroPadding2D((1,1))(temp_aggr)
    output = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(128, (3, 3), activation='relu', name='conv2_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(128, (3, 3), activation='relu', name='conv2_2')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(256, (3, 3), activation='relu', name='conv3_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv4_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_1')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_2')(output)
    output = ZeroPadding2D((1,1))(output)
    output = Convolution2D(512, (3, 3), activation='relu', name='conv5_3')(output)
    output = MaxPooling2D((2,2), strides=(2,2))(output)

    inputs = [frame1, frame2, frame3, frame4, frame5]
    model = Model(inputs=inputs, outputs=output)
    
    # load VGG-face weigths
    model.load_weights(vgg_weights_path, by_name=True)
    for layer in ['conv1_1a', 'conv1_1b', 'conv1_1c', 'conv1_1d', 'conv1_1e']:
        model.get_layer(layer).set_weights(conv1_1_weigths)

    return model

def get_conv_1_1_weights(vgg_weights_path):
    """Returns the weigths of first convolutional layer of VGG-Face (conv_1_1).
    """
    temp_mod = Sequential()
    temp_mod.add(ZeroPadding2D((1,1),input_shape=(img_size, img_size, nb_channels)))
    temp_mod.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    temp_mod.load_weights(vgg_weights_path, by_name=True)
    conv1_1_weigths = temp_mod.get_layer('conv1_1').get_weights()
    return conv1_1_weigths

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

def train_phrnn_crossval(create_model, landmarks_inputs, y, batch_size, epochs, callbacks, n_splits, save_best_model=False, model_path=None):
    """Train PHRNN model using n-fold crossvalidation.
    """
    # Prepare for cross-validation
    labels = np.argmax(y, axis=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

    histories = []
    val_accs = []
    for i, (train, test) in enumerate(skf.split(landmarks_inputs[0], labels)):
        print("Running Fold {}/{}".format(i+1, n_splits))
        model = create_model()
        
        if save_best_model:
            cur_model_path = model_path[:-3]+'_split'+str(i+1)+model_path[-3:]
            model_checkpoint = ModelCheckpoint(cur_model_path, monitor='val_loss', save_best_only=True)
            cur_callbacks = callbacks.copy()
            cur_callbacks.append(model_checkpoint)

        hist = model.fit([x[train] for x in landmarks_inputs],
                         y[train],
                         validation_data = ([x[test] for x in landmarks_inputs], y[test]),
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         callbacks=cur_callbacks if save_best_model else callbacks,
                         verbose=0)
        histories.append(hist)

        # Save the val_acc of the best epoch
        best_epoch = np.argmin(hist.history['val_loss'])
        best_val_acc = hist.history['val_acc'][best_epoch]
        val_accs.append(best_val_acc)
        print("  Best model : epoch {} - val_acc: {:.6f}".format(best_epoch+1, best_val_acc))

    if save_best_model:
        model = load_model(model_path[:-3]+'_split1'+model_path[-3:])

    cross_val_acc = np.mean(val_accs)
    print('Average validation accuracy : {:.6f}'.format(cross_val_acc))
    
    return model, skf, histories
    
def create_tcnn_top():
    """Create the top of the tcnn with fully connected layers.
    """
    input_shape=(7, 7, 512)

    tcnn_top = Sequential()
    tcnn_top.add(Convolution2D(1024, (7, 7), activation='relu', name='fc6', input_shape=input_shape))
    tcnn_top.add(Dropout(0.5))
    tcnn_top.add(Convolution2D(512, (1, 1), activation='relu', name='fc7'))
    tcnn_top.add(Dropout(0.5))
    tcnn_top.add(Convolution2D(7, (1, 1), name='fc8'))
    tcnn_top.add(Flatten())
    tcnn_top.add(Activation('softmax'))
    
    tcnn_top.compile(loss='categorical_crossentropy',
                 optimizer=optimizers.Adam(),
                 metrics=['accuracy'])
    
    return tcnn_top

def train_tcnn_crossval(create_model, x, y, batch_size, epochs, callbacks, n_splits, save_best_model=False, model_path=None):
    """Train the TCNN model using n-fold crossvalidation.
    """
    # Prepare for cross-validation
    labels = np.argmax(y, axis=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

    histories = []
    val_accs = []
    for i, (train, test) in enumerate(skf.split(x, labels)):
        print("Running Fold {}/{}".format(i+1, n_splits))
        model = create_model()
        
        if save_best_model:
            cur_model_path = model_path[:-3]+'_split'+str(i+1)+model_path[-3:]
            model_checkpoint = ModelCheckpoint(cur_model_path, monitor='val_loss', save_best_only=True)
            cur_callbacks = callbacks.copy()
            cur_callbacks.append(model_checkpoint)

        hist = model.fit(x[train],
                         y[train],
                         validation_data = (x[test], y[test]),
                         batch_size=batch_size,
                         epochs=epochs,
                         shuffle=True,
                         callbacks=cur_callbacks if save_best_model else callbacks,
                         verbose=0)
        histories.append(hist)

        # Save the val_acc of the best epoch
        best_epoch = np.argmin(hist.history['val_loss'])
        best_val_acc = hist.history['val_acc'][best_epoch]
        val_accs.append(best_val_acc)
        print("  Best model : epoch {} - val_acc: {:.6f}".format(best_epoch+1, best_val_acc))

    if save_best_model:
        model = load_model(model_path[:-3]+'_split1'+model_path[-3:])

    cross_val_acc = np.mean(val_accs)
    print('Average validation accuracy : {:.6f}'.format(cross_val_acc))
    
    return model, skf, histories

def load_cross_val_models(tcnn_model_path, phrnn_model_path, model_path_ext, n_splits, y):
    """Load models from each split of the cross-validation.
    """
    # Get back the train/test split used
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    labels = np.argmax(y, axis=1)
    train_test = [(train, test) for (train,test) in skf.split(y, labels)]
    train_idx, test_idx = zip(*train_test)

    tcnn_models_paths = [tcnn_model_path+'_split'+str(i+1)+model_path_ext for i in range(len(test_idx))]
    phrnn_models_paths = [phrnn_model_path+'_split'+str(i+1)+model_path_ext for i in range(len(test_idx))]

    print('Loading TCNN models for cross-validation...')
    tcnn_models = [load_model(path) for path in tcnn_models_paths]
    print('Loading PHRNN models for cross-validation...')
    phrnn_models = [load_model(path) for path in phrnn_models_paths]
    
    return tcnn_models, phrnn_models

def cross_validate_tcnn_phrnn(merge_weights, n_splits, tcnn_models, phrnn_models, vgg_features, landmarks_inputs, y):
    # Get back the train/test split used
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    labels = np.argmax(y, axis=1)
    train_test = [(train, test) for (train,test) in skf.split(y, labels)]
    train_idx, test_idx = zip(*train_test)

    accuracies = np.zeros((len(test_idx), len(merge_weights)))

    for i in range(len(test_idx)):
        test_indices = test_idx[i]

        tcnn_top = tcnn_models[i]
        phrnn = phrnn_models[i]

        # True emotions
        y_true = np.argmax(y[test_indices], axis=1)

        # TCNN emotion prediction
        y_pred_tcnn = tcnn_top.predict(vgg_features[test_indices])

        # PHRNN emotion prediction
        y_pred_phrnn = phrnn.predict([x[test_indices] for x in landmarks_inputs])

        for (j, w) in enumerate(merge_weights):
            y_pred_merged = w*y_pred_tcnn + (1-w)*y_pred_phrnn
            y_pred_classes = np.argmax(y_pred_merged, axis=1)

            acc = (y_true == y_pred_classes).sum()/len(y_true)
            accuracies[i,j] = acc

    return accuracies
