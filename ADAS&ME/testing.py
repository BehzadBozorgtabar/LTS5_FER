import os
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils.np_utils import to_categorical

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

from training import leave_one_out_split, DataGenerator, get_labels_from_annotations, get_file_annotations, get_files_list, get_labels_from_pickle
from const import *

# def get_labels_from_annotation(annotations_file_path, nb_frames_per_sample):
#     # Load annotations from file
#     annotations = pd.read_csv(annotations_file_path)
#     y = annotations.apply(lambda row: int(row['Severity'])-1, axis=1).values
#     trim = y.shape[0] - y.shape[0]%nb_frames_per_sample
#     y = y[:trim].reshape((-1, nb_frames_per_sample))
#     y = to_categorical(y, num_classes=nb_emotions) 
#     y = np.mean(y, axis=1)
#     y = np.argmax(y, axis=1)
#     return y

def evaluate_model(model_path, data_path, annotations_path, files_list, files_per_batch, nb_frames_per_sample=5):
    y_pred = np.array([])
    y_true = np.array([])
    
    subjects_list = sorted(list(set([subj for (subj,f) in files_list])))
    for i, subject in enumerate(subjects_list):
        print(" Testing on {}...".format(subject))
        
        # Load model of the current split 
        cur_model_path = model_path[:-3]+'_'+subject+model_path[-3:]
        model = load_model(cur_model_path)
        
        train_files_list, val_files_list = leave_one_out_split(subject, files_list)
        validation_generator = DataGenerator(val_files_list, files_per_batch=files_per_batch, data_path=data_path, annotations_path=annotations_path)
        
        cur_y_pred = model.predict_generator(generator=validation_generator)
        cur_y_pred = np.argmax(cur_y_pred, axis=1)
        y_pred = np.concatenate([y_pred, cur_y_pred])
        
        cur_y_true = []
        for (s, file) in val_files_list:
            annotations_file_path = annotations_path+'/'+s+'/'+file+'_annotated.csv'
            file_y_true = get_labels_from_annotation(annotations_file_path, nb_frames_per_sample=5)
            
            if s is 'TS11_DRIVE' and file is '20180827_115840':
                file_y_true = file_y_true[:14250//nb_frames_per_sample]
            cur_y_true.append(file_y_true)
        
        cur_y_true = np.concatenate(cur_y_true)
        y_true = np.concatenate([y_true, cur_y_true])
        
        print(y_pred.shape, y_true.shape)

    #print_model_eval_metrics(y_pred, y_true)
    
    return y_pred, y_true

def evaluate_tcnn_phrnn_model(tcnn_model_path, phrnn_model_path, phrnn_features, subjects_list, data_path, frames_data_path, files_per_batch=1, nb_frames_per_sample=5, merge_weight=0.5):
    y_pred = np.array([])
    y_true = np.array([])
    
    files_list = get_files_list(subjects_list, frames_data_path)
    for i, subject in enumerate(subjects_list):
        print("Testing on {}...".format(subject))
        
        train_files_list, val_files_list = leave_one_out_split(subject, files_list)
        tcnn_val_generator = DataGenerator(val_files_list, data_path, frames_data_path, files_per_batch=files_per_batch, features='vgg-tcnn')
        phrnn_val_generator = DataGenerator(val_files_list, data_path, frames_data_path, files_per_batch=files_per_batch, features=phrnn_features)
        
        # Load models of the current split 
        if tcnn_model_path is not None:
            cur_tcnn_model_path = tcnn_model_path[:-3]+'_'+subject+tcnn_model_path[-3:]
            tcnn_model = load_model(cur_tcnn_model_path)
            y_pred_tcnn = tcnn_model.predict_generator(generator=tcnn_val_generator)
        if phrnn_model_path is not None:
            cur_phrnn_model_path = phrnn_model_path[:-3]+'_'+subject+phrnn_model_path[-3:]
            phrnn_model = load_model(cur_phrnn_model_path)
            y_pred_phrnn = phrnn_model.predict_generator(generator=phrnn_val_generator)
        
        if (tcnn_model_path is not None) and (phrnn_model_path is not None):
            cur_y_pred = merge_weight*y_pred_tcnn + (1-merge_weight)*y_pred_phrnn
        elif tcnn_model_path is None:
            cur_y_pred = y_pred_phrnn
        else:
            cur_y_pred = y_pred_tcnn
        
        cur_y_pred = np.argmax(cur_y_pred, axis=1)
        y_pred = np.concatenate([y_pred, cur_y_pred])
        
        cur_y_true = []
        for (s, file) in val_files_list:
            pickle_file_path = frames_data_path+'/'+s+'/'+file+'.pkl'
            file_y_true, _, _ = get_labels_from_pickle(pickle_file_path)
            
            if s is 'TS11_DRIVE' and file is '20180827_115840':
                file_y_true = file_y_true[:14250//nb_frames_per_sample]
            cur_y_true.append(file_y_true)
        
        cur_y_true = np.concatenate(cur_y_true)
        y_true = np.concatenate([y_true, cur_y_true])
        
        print(y_pred.shape, y_true.shape)

    #print_model_eval_metrics(y_pred, y_true)
    return y_pred, y_true

def print_model_eval_metrics(y_pred, y_true, average = 'macro'):
    acc = (y_true == y_pred).sum()/len(y_pred)

    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    print('\nMODEL EVALUATION:')
    print('  Accuracy  : {:.5f}'.format(acc))
    print('  Precision : {:.5f}'.format(precision))
    print('  Recall    : {:.5f}'.format(recall))
    print('  F1-score  : {:.5f}'.format(f1))