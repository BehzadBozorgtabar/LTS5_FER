import numpy as np
import pickle
import pandas as pd
import os

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
        annotations = pd.concat(annotations)

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
    mask1 = (y!=-999).all(axis=1)
    y = y[mask1]

    # Discard samples containing different emotions
    mask2 = [len(set(s))<=1 for s in y]
    y = y[mask2]
    
    # Take emotion of first frame
    y = y[:,0]
    
    valid_mask = np.logical_and(mask1, np.array(mask2))
    return y, valid_mask

class DataGenerator(Sequence):
    'Generates data for training on the ADAS&ME dataset'
    def __init__(self, files_list, files_per_batch=1, features='sift-vgg', data_path='data',
                 annotations_path='ADAS&ME_data/annotated', nb_frames_per_sample=5):
        'Initialization'
        self.files_list = files_list
        self.files_per_batch = files_per_batch
        self.features = features
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.nb_frames_per_sample = nb_frames_per_sample

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files_list) / self.files_per_batch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        files = self.files_list[index*self.files_per_batch:(index+1)*self.files_per_batch]

        # Generate data
        X, y = self.__data_generation(files)

        return X, y
    
    def load_data_from_file(self, file):
        'Loads the data from a the given file'
        annotations = get_file_annotations(annotations_path, file)
        y, valid_mask = get_labels_from_annotations(annotations)
        
        sift_data_path = self.data_path+'/'+file[0]+'/sift_'+file[1]+'.pkl'
        vgg_data_path = self.data_path+'/'+file[0]+'/vggCustom_'+file[1]+'.pkl'
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
            sift.shape

            # Prepare the different landmarks clusters
            eyebrows_landmarks = sift[:,:, eyebrows_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            nose_landmarks = sift[:,:, nose_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            eyes_landmarks = sift[:,:, eyes_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            inside_lip_landmarks = sift[:,:, inside_lip_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))
            outside_lip_landmarks = sift[:,:, outside_lip_mask].reshape((sift.shape[0], self.nb_frames_per_sample, -1))

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
        
        return x, y

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        if self.features == 'sift-phrnn':
            return self.load_data_from_file(list_files_temp[0])
        else:
            x = []
            y = []

            for file in list_files_temp:
                x_file, y_file = self.load_data_from_file(file)
                x.append(x_file)
                y.append(y_file)

            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)

            return x, y

def leave_one_out_split(test_subj, files_list):
    """Returns the train/test split for subject-wise leave-one-out.
    """
    train_files_list = [(subj,f) for (subj,f) in files_list if subj!=test_subj]
    val_files_list = [(subj,f) for (subj,f) in files_list if subj==test_subj]
    
    return train_files_list, val_files_list

def train_leave_one_out(create_model, data_path, features, annotations_path, files_list, files_per_batch, epochs, callbacks, class_weight=None, save_best_model=False, save_each_split=True, model_path=None):
    """Train a model using leave-one-out.
    """
    if save_best_model and not save_each_split:
        cur_callbacks = callbacks.copy()
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        cur_callbacks.append(model_checkpoint)

    histories = []
    val_accs = []
    subjects_list = sorted(list(set([subj for (subj,f) in files_list])))
    for i, subject in enumerate(subjects_list):
        print("Running Fold {}/{} : testing on {}".format(i+1, len(subjects_list), subject))
        model = create_model()

        if save_best_model and save_each_split:
            cur_model_path = model_path[:-3]+'_'+subject+model_path[-3:]
            model_checkpoint = ModelCheckpoint(cur_model_path, monitor='val_loss', save_best_only=True)
            cur_callbacks = callbacks.copy()
            cur_callbacks.append(model_checkpoint)
        
        train_files_list, val_files_list = leave_one_out_split(subject, files_list)
        training_generator = DataGenerator(train_files_list, nb_frames_per_sample=5, files_per_batch=files_per_batch, features=features, data_path=data_path, annotations_path=annotations_path)
        validation_generator = DataGenerator(val_files_list, nb_frames_per_sample=5, files_per_batch=files_per_batch, features=features, data_path=data_path, annotations_path=annotations_path)
        
        hist = model.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    epochs=epochs,
                                    class_weight=class_weight,
                                    use_multiprocessing=True,
                                    workers=6,
                                    callbacks=cur_callbacks if save_best_model else callbacks,
                                    verbose=1)
        histories.append(hist)

        # Save the val_acc of the best epoch
        best_epoch = np.argmin(hist.history['val_loss'])
        best_val_loss = hist.history['val_loss'][best_epoch]
        best_val_acc = hist.history['val_acc'][best_epoch]
        val_accs.append(best_val_acc)
        print("  Best model : epoch {} - val_loss: {:.6f} - val_acc: {:.6f}".format(best_epoch, best_val_loss, best_val_acc))

    cross_val_acc = np.mean(val_accs)
    print('Average validation accuracy : {:.6f}'.format(cross_val_acc))

    return model, histories


def get_class_weight(files_list, annotations_path, mode='balanced'):
    """Returns the class weights computed from the class distribution.
    """
    counts = [0]*nb_emotions
    y = []
    for subj, file in files_list:
        annotations = pd.read_csv(annotations_path+'/'+subj+'/'+file+"_annotated.csv")
        y_file = annotations.apply(lambda row: int(row['Severity'])-1, axis=1)
        y = np.concatenate([y, y_file])

    weights = compute_class_weight(mode, list(range(nb_emotions)), y)
    class_weight = {}
    for i, w in enumerate(weights):
        class_weight[i] = w
    return class_weight#, Counter(y)
