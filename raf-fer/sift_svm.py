import sys
import os
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from extraction import extract_data, extract_all_sift_features
from const import *

# Train the model only if the call was made with argument 'train',
# otherwise, just test it
train =  len(sys.argv) > 1 and sys.argv[1]=='train'

# If the sift features were pre-computed, we don't have to recompute them
load_sift_features = os.path.isfile(sift_features_path)


#### EXTRACTION ####

if not load_sift_features:
	# Extract data from raw dataset
	x_train, x_test, y_train, y_test = extract_data(images_path, label_path)
else:
	with open(label_data_path, 'rb') as f:
		y_train, y_test = pickle.load(f)


# Compute SIFT features or loads them if available
if load_sift_features:
	with open(sift_features_path, 'rb') as f:
		sift_features_train, sift_features_test = pickle.load(f)
else:
	sift_features_train = extract_all_sift_features(x_train)
	sift_features_test = extract_all_sift_features(x_test)


#### PRE-PROCESSING ####

# Standardization and PCA on the SIFT features
std = StandardScaler()
pca = PCA(n_components=1024, svd_solver='randomized', whiten=True)
preprocess = make_pipeline(std, pca)
preprocess.fit(sift_features_train)
print('Doing PCA on SIFT features - Total explained variance: {:.2f}%'.format(100*sum(pca.explained_variance_ratio_)))

sift_train_pca = preprocess.transform(sift_features_train)
sift_test_pca = preprocess.transform(sift_features_test)


#### TRAINING ####

if train:
	print('Training the SVM model...')
	sift_svm = SVC(C=5, gamma=0.0003, kernel='rbf', class_weight='balanced', verbose=False, random_state=42)
	sift_svm.fit(sift_train_pca, y_train)
	print('Training complete.')
else:
	with open(sift_svm_model_path, 'rb') as f:
		sift_svm = pickle.load(f)


#### TESTING ####

# Get predictions of test set
print('Computing predictions on test set...')
y_pred = sift_svm.predict(sift_test_pca)

# Computes the accuracy
acc = (y_pred==y_test).sum()/len(y_pred)
print('Test accuracy : {:.4f}'.format(acc))
