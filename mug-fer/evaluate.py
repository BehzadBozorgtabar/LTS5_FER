import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from keras.models import load_model

def evaluate_model(x, y, model_path, n_splits):
    labels = np.argmax(y, axis=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    y_pred = np.array([])
    y_true = np.array([])
    accuracies = []

    for i, (train, test) in enumerate(skf.split(x, labels)):
        # Load model of the current split 
        cur_model_path = model_path[:-3]+'_split'+str(i+1)+model_path[-3:]
        model = load_model(cur_model_path)
        
        # Predict emotion of current test set
        cur_y_pred = model.predict_classes(x[test])
        y_pred = np.concatenate([y_pred, cur_y_pred])
        
        y_true = np.concatenate([y_true, labels[test]])
    
    print_model_eval_metrics(y_pred, y_true)
    
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
