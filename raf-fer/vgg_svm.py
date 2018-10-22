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

from extraction import extract_data, get_vgg_custom
from const import *

# Train the model only if the call was made with argument 'train',
# otherwise, just test it
train =  len(sys.argv) > 1 and sys.argv[1]=='train'

# If the sift or vgg features were pre-computed, we don't have to recompute them
load_vgg_features = os.path.isfile(vgg_fc6_path)


#### EXTRACTION ####

if not load_vgg_features:
	# Extract data from raw dataset
	x_train, x_test, y_train, y_test = extract_data(images_path, label_path)
else:
	with open(label_data_path, 'rb') as f:
		y_train, y_test = pickle.load(f)


if load_vgg_features:
	with open(vgg_fc6_path, 'rb') as f:
		vgg_fc6_train, vgg_fc6_test = pickle.load(f)
else:
	# Create fc6 VGG model
	vgg_fc6 = get_vgg_custom(custom_vgg_weights_path, output_layer='fc6')

	# Compute VGG fc6 features or loads them if available
	print('Extracting VGG features...')
	vgg_fc6_train = vgg_fc6.predict(x_train)
	vgg_fc6_test = vgg_fc6.predict(x_test)


#### PRE-PROCESSING ####

# Standardization of the VGG features
std = StandardScaler()
std.fit(vgg_fc6_train)
vgg_train_std = std.transform(vgg_fc6_train)
vgg_test_std = std.transform(vgg_fc6_test)


#### TRAINING ####

if train:
	print('Training the SVM model...')
	vgg_svm = SVC(C=1, gamma=0.0003, kernel='rbf', class_weight='balanced', cache_size=1000, verbose=False)
	vgg_svm.fit(vgg_train_std, y_train)
	print('Training complete.')
else:
	with open(vgg_svm_model_path, 'rb') as f:
		vgg_svm = pickle.load(f)


#### TESTING ####

# Get predictions of test set
print('Computing predictions on test set...')
y_pred = vgg_svm.predict(vgg_test_std)

# Computes the accuracy
acc = (y_pred==y_test).sum()/len(y_pred)
print('Test accuracy : {:.4f}'.format(acc))
