import numpy as np
import pickle
import pandas as pd
import os
from collections import Counter
import math

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, SimpleRNN, Bidirectional, LSTM, BatchNormalization, GlobalAveragePooling2D, average, concatenate
from keras.callbacks import Callback, ModelCheckpoint
from keras import optimizers
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from sklearn.utils.class_weight import compute_class_weight

from const import *

class CustomVerbose(Callback):
    """Callback that prints a short 1-line verbose for each epoch and overwrites over the past one.
    """
    def __init__(self, epochs):
        self.epochs = epochs
        
    def on_train_end(self, logs={}):
        print('\n')
        
    def on_epoch_end(self, epoch, dictionary):        
        print('  Epoch %d/%d'%(epoch+1, self.epochs), ''.join([' - %s: %.6f'%tup for tup in dictionary.items()]), end="\r")

def normalize_face_landmarks(face_landmarks):
    """Normalize facial landmarks by subtracting the coordinates corresponding
       to the nose, and dividing by the standard deviation.
    """
    face_landmarks_norm = np.zeros(face_landmarks.shape)
    
    for (i, lm) in enumerate(face_landmarks):
        face_landmarks_norm[i] = lm - lm[nose_center_idx]
            
    std_x = np.std(face_landmarks_norm[:,:,0].reshape((-1,)))
    std_y = np.std(face_landmarks_norm[:,:,1].reshape((-1,)))
    
    face_landmarks_norm[:,:,0] = np.multiply(face_landmarks_norm[:,:,0], 1./std_x)
    face_landmarks_norm[:,:,1] = np.multiply(face_landmarks_norm[:,:,1], 1./std_y)
    
    return face_landmarks_norm

def get_file_annotations(annotations_path, file):
    """Returns a dataframe containing all annotations for a given file.
    """
    annotations_file_path = annotations_path+'/'+file[0]+'/'+file[1]+'_annotated{}.csv'
    if os.path.isfile(annotations_file_path.format('')) :
        # if file contains less than 1000 frames, annotations are stored in a single csv file
        annotations = pd.read_csv(annotations_file_path.format(''))
    else:
        # otherwise, stored by chunks of 1000
        i = 1
        annotations = []
        while os.path.isfile(annotations_file_path.format(i)):
            annot = pd.read_csv(annotations_file_path.format(i))
            annotations.append(annot)
            i += 1
        annotations = pd.concat(annotations, sort=True)
        
    return annotations

def get_labels_from_annotations(annotations, nb_frames_per_sample=5):
    """Returns the (filtered) labels for the given file, 
    as well as the mask applied during filtering.
    """
    y = annotations.apply(lambda row: int(row['Severity'])-1, axis=1).values

    # group by 'nb_frames_per_sample' frames
    trim = y.shape[0] - y.shape[0]%nb_frames_per_sample
    y = y[:trim].reshape((-1, nb_frames_per_sample))

    # Discard samples containing invalid annotation '-999'
    mask1 = (y>=0).all(axis=1)

    # Discard samples containing different emotions
    mask2 = np.array([len(set(s))<=1 for s in y])
    
    # Discard samples that contain frames from different cameras
    camera_indices = annotations.apply(lambda row: int(row['CameraIndex']), axis=1).values
    camera_indices = camera_indices[:trim].reshape((-1, nb_frames_per_sample))
    mask3 = np.array([len(set(i))<=1 for i in camera_indices])
    
    valid_mask = np.logical_and(mask1, np.logical_and(mask2, mask3))
    y = y[valid_mask]
    
    # Take emotion of first frame
    y = y[:,0]
    
    # Discard samples that contain frames from different cameras
    frame_numbers = annotations.apply(lambda row: int(row['FrameNumber']), axis=1).values
    frame_numbers = frame_numbers[:trim].reshape((-1, nb_frames_per_sample))[valid_mask]
    
    return y, valid_mask, frame_numbers

def get_labels_from_pickle(pickle_file_path, emo_conv=None, nb_frames_per_sample=5):
    """Returns the (filtered) labels for the given file, 
    as well as the mask applied during filtering.
    """
    with open(pickle_file_path, 'rb') as f:
        _, roi =  pickle.load(f)
    
    if emo_conv is None:
        y = np.array([r['Severity']-1 for r in roi])
    else:
        y = np.array([emo_conv[r['Severity']-1] for r in roi])
    camera_idx = np.array([r['CameraIndex'] for r in roi])
    frames_nbs = np.array([r['FrameNumber'] for r in roi])
    
    # group by 'nb_frames_per_sample' frames
    trim = y.shape[0] - y.shape[0]%nb_frames_per_sample
    y = y[:trim].reshape((-1, nb_frames_per_sample))

    # Discard samples containing invalid annotation '-999'
    mask1 = (y>=0).all(axis=1)

    # Discard samples containing different emotions
    mask2 = np.array([len(set(s))<=1 for s in y])
    
    # Discard samples that contain frames from different cameras
    camera_idx = camera_idx[:trim].reshape((-1, nb_frames_per_sample))
    mask3 = np.array([len(set(i))<=1 for i in camera_idx])
    
    # Discard that have non-consecutive frames
    frames_nbs = frames_nbs[:trim].reshape((-1, nb_frames_per_sample))
    mask4 = np.array([abs(fr_nbs[-1]-fr_nbs[0])<=4 for fr_nbs in frames_nbs])
    
    valid_mask = np.logical_and(mask1, np.logical_and(mask2, np.logical_and(mask3, mask4)))
    y = y[valid_mask]
    
    # Take emotion of first frame
    y = y[:,0]
    
    frames_nbs = frames_nbs[valid_mask]
    
    return y, valid_mask, frames_nbs

def get_files_list(subjects, frames_data_path):
    files_list = []
    for subj in subjects:
        subj_frames_data_path = frames_data_path+'/'+subj
        files = sorted([(subj, f[:-4]) for f in os.listdir(subj_frames_data_path) if f.endswith('.pkl')])
        files_list += files
    return files_list

class DataGenerator(Sequence):
    'Generates data for training on the ADAS&ME dataset'
    def __init__(self, files_list, data_path, frames_data_path, files_per_batch=1, features='sift-vgg', nb_frames_per_sample=5, shuffle=True):
        'Initialization'
        self.files_list = files_list
        self.data_path = data_path
        self.frames_data_path = frames_data_path
        self.files_per_batch = files_per_batch
        self.features = features
        self.nb_frames_per_sample = nb_frames_per_sample
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files_list) / self.files_per_batch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.files_indexes[index*self.files_per_batch:(index+1)*self.files_per_batch]

        # Generate indexes of the batch
        files = [self.files_list[i] for i in indexes]
        #print(files)
        # Generate data
        X, y, frame_numbers = self.__data_generation(files)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.files_indexes = np.arange(len(self.files_list))
        if self.shuffle == True:
            np.random.shuffle(self.files_indexes)
    
    def load_data_from_file(self, file):
        'Loads the data from a the given file'
        pickle_file_path = self.frames_data_path+'/'+file[0]+'/'+file[1]+'.pkl'
        y, valid_mask, frame_numbers = get_labels_from_pickle(pickle_file_path, emo_conv)
        y = to_categorical(y, num_classes=nb_emotions) 
        
        sift_data_path = self.data_path+'/'+file[0]+'/sift/sift_'+file[1]+'.pkl'
        landmarks_data_path = self.data_path+'/'+file[0]+'/landmarks/landmarks_'+file[1]+'.pkl'
        vgg_data_path = self.data_path+'/'+file[0]+'/vgg/vggCustom_'+file[1]+'.pkl'
        vgg_tcnn_data_path = self.data_path+'/'+file[0]+'/vgg_tcnn/vgg_tcnn_'+file[1]+'.pkl'
        
        if self.features == 'vgg-sift' or self.features == 'sift-vgg':
            with open(sift_data_path, 'rb') as f:
                sift_features = pickle.load(f)
            sift_features = sift_features.reshape(sift_features.shape[0], -1)

            with open(vgg_data_path, 'rb') as f:
                vgg_features = pickle.load(f)

            # Concatenate the VGG features with the SIFT features
            vgg_sift_features = np.concatenate([vgg_features, sift_features], axis=1)
            trim = vgg_sift_features.shape[0] - vgg_sift_features.shape[0]%self.nb_frames_per_sample
            vgg_sift_features = vgg_sift_features[:trim]
            x = vgg_sift_features.reshape((-1, self.nb_frames_per_sample, vgg_sift_features.shape[1]))
            
            # Filter out samples whose annotations are invalid
            x = x[valid_mask]
            
        elif self.features == 'sift-phrnn':
            with open(sift_data_path, 'rb') as f:
                sift_features = pickle.load(f)
                
            trim = sift_features.shape[0] - sift_features.shape[0]%self.nb_frames_per_sample
            sift = sift_features[:trim]
            sift = sift.reshape((-1, self.nb_frames_per_sample, 51, 128))

            # Prepare the different landmarks clusters
            eyebrows_landmarks = sift[:,:, eyebrows_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            nose_landmarks = sift[:,:, nose_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            eyes_landmarks = sift[:,:, eyes_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            inside_lip_landmarks = sift[:,:, inside_lip_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            outside_lip_landmarks = sift[:,:, outside_lip_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))

            x = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]
            
            # Filter out samples whose annotations are invalid
            x = [xt[valid_mask] for xt in x]
            
        elif self.features == 'landmarks-phrnn':
            with open(landmarks_data_path, 'rb') as f:
                landmarks = pickle.load(f)
                
            landmarks_norm = normalize_face_landmarks(landmarks)

            trim = landmarks_norm.shape[0] - landmarks_norm.shape[0]%self.nb_frames_per_sample
            lm = landmarks_norm[:trim]
            lm = lm.reshape((-1, self.nb_frames_per_sample, 51, 2))

            # Prepare the different landmarks clusters
            eyebrows_landmarks = lm[:,:, eyebrows_mask].reshape((lm.shape[0], self.nb_frames_per_sample, -1))
            nose_landmarks = lm[:,:, nose_mask].reshape((lm.shape[0], self.nb_frames_per_sample, -1))
            eyes_landmarks = lm[:,:, eyes_mask].reshape((lm.shape[0], self.nb_frames_per_sample, -1))
            inside_lip_landmarks = lm[:,:, inside_lip_mask].reshape((lm.shape[0], self.nb_frames_per_sample, -1))
            outside_lip_landmarks = lm[:,:, outside_lip_mask].reshape((lm.shape[0], self.nb_frames_per_sample, -1))

            x = [eyebrows_landmarks, nose_landmarks, eyes_landmarks, inside_lip_landmarks, outside_lip_landmarks]
            
            # Filter out samples whose annotations are invalid
            x = [xt[valid_mask] for xt in x]

        elif self.features == 'vgg-tcnn':
            with open(vgg_tcnn_data_path, 'rb') as f:
                vgg_tcnn_features = pickle.load(f)
            x = vgg_tcnn_features
            
            # Filter out samples whose annotations are invalid
            x = x[valid_mask]
            
        else:
            raise ValueError('\'features\' parameter is invalid.')
        
        return x, y, frame_numbers

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        #if 'phrnn' in self.features:
         #   return self.load_data_from_file(list_files_temp[0])
        #else:
        x = []
        y = []

        for file in list_files_temp:
            x_file, y_file, frame_numbers = self.load_data_from_file(file)
            x.append(x_file)
            y.append(y_file)
        
        if 'phrnn' in self.features:
            x_concat = []
            for clstr_idx in range(5): # corresponding to the 5 clusters of landmarks
                xt = [x[i][clstr_idx] for i in range(len(x))]
                xt = np.concatenate(xt, axis=0)
                x_concat.append(xt)
            x = x_concat
        else:
            x = np.concatenate(x, axis=0)
            
        y = np.concatenate(y, axis=0)

        return x, y, frame_numbers
        
    def load_all_data(self):
        x, y, _ = self.__data_generation(self.files_list)
        return x, y
    

def leave_one_out_split(test_subj, files_list):
    """Returns the train/test split for subject-wise leave-one-out.
    """
    train_files_list = [(subj,f) for (subj,f) in files_list if subj!=test_subj]
    val_files_list = [(subj,f) for (subj,f) in files_list if subj==test_subj]
    
    return train_files_list, val_files_list

def train_leave_one_out(create_model, data_path, frames_data_path, subjects, features, epochs, callbacks, files_per_batch=1, class_weight=None, save_best_model=False, save_each_split=True, model_path=None):
    """Train a model using leave-one-out.
    """
    if save_best_model and not save_each_split:
        cur_callbacks = callbacks.copy()
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        cur_callbacks.append(model_checkpoint)

    histories = []
    val_accs = []
    files_list = get_files_list(subjects, frames_data_path)
    subjects_list = sorted(list(set([subj for (subj,f) in files_list])))

    # Files starting with S should not be used for testing
    subjects_list_test = [subj for subj in subjects_list if not subj.startswith('S')]

    for i, subject in enumerate(subjects_list_test):
        print("Running Fold {}/{} : testing on {}".format(i+1, len(subjects_list_test), subject))
        model = create_model()

        if save_best_model and save_each_split:
            cur_model_path = model_path[:-3]+'_'+subject+model_path[-3:]
            model_checkpoint = ModelCheckpoint(cur_model_path, monitor='val_loss', save_best_only=True)
            cur_callbacks = callbacks.copy()
            cur_callbacks.append(model_checkpoint)

        print('Computing class weights...')
        train_subj_list = subjects.copy()
        train_subj_list.remove(subject)
        class_weight = get_class_weight(train_subj_list, frames_data_path, emo_conv)
        print(class_weight)
        
        train_files_list, val_files_list = leave_one_out_split(subject, files_list)

        training_generator = DataGenerator(train_files_list, data_path, frames_data_path, files_per_batch=files_per_batch, features=features)
        x_train, y_train = training_generator.load_all_data()

        validation_generator = DataGenerator(val_files_list, data_path, frames_data_path, files_per_batch=files_per_batch, features=features)
        x_val, y_val = validation_generator.load_all_data()

        hist = model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        class_weight=class_weight,
                        callbacks=cur_callbacks if save_best_model else callbacks,
                        verbose=1)

        histories.append(hist)

        # Save the val_acc of the best epoch
        best_epoch = np.argmin(hist.history['val_loss'])
        best_val_loss = hist.history['val_loss'][best_epoch]
        best_val_acc = hist.history['val_acc'][best_epoch]
        val_accs.append(best_val_acc)
        print("  Best model : epoch {} - val_loss: {:.6f} - val_acc: {:.6f}".format(best_epoch+1, best_val_loss, best_val_acc))

    cross_val_acc = np.mean(val_accs)
    print('Average validation accuracy : {:.6f}'.format(cross_val_acc))

    return model, histories

def get_class_weight(subjects, frames_data_path, emo_conv=None, mode='balanced', power=1):
    """Returns the class weights computed from the class distribution.
    """
    files_list = get_files_list(subjects, frames_data_path)
    counts = [0]*nb_emotions
    y = []
    for subj, file in files_list:
        #annotations = pd.read_csv(annotations_path+'/'+subj+'/'+file+"_annotated.csv")
        pickle_file_path = frames_data_path+'/'+subj+'/'+file+'.pkl'
        y_file, _, _ = get_labels_from_pickle(pickle_file_path, emo_conv)

        y = np.concatenate([y, y_file])
        
    print("Class distribution : "+str(Counter(y)))
    
    weights = compute_class_weight(mode, list(range(nb_emotions)), y)
    weights = [math.pow(w, power) for w in weights]

    class_weight = {}
    for i, w in enumerate(weights):
        class_weight[i] = w
    return class_weight

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
                     optimizer=optimizers.Adam(),#lr=0.001
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
                      optimizer=optimizers.Adam(),#lr=0.001
                      metrics=['accuracy'])
        return phrnn
    
    return phrnn_creator
