import numpy as np
from sklearn.cross_validation import StratifiedKFold

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, TimeDistributed, LSTM, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import Callback

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


def train_crossval(create_model, x, y, batch_size, epochs, callbacks, n_folds, save_best_model=False, weights_path=None):
	"""Train a model using n-folds crossvalidation.
	"""
	# Prepare for cross-validation
	labels = np.argmax(y, axis=1)
	skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=False)

	histories = []
	val_accs = []
	for i, (train, test) in enumerate(skf):
		print("Running Fold {}/{}".format(i+1, n_folds))
		model = create_model()

		hist = model.fit(x[train],
						 y[train],
						 validation_data = (x[test], y[test]),
						 batch_size=batch_size,
						 epochs=epochs,
						 shuffle=True,
						 callbacks=callbacks,
						 verbose=0)
		histories.append(hist)

		# Save the val_acc of the best epoch
		best_epoch = np.argmax(hist.history['val_acc'])
		best_val_acc = hist.history['val_acc'][best_epoch]
		val_accs.append(best_val_acc)
		print("  Best model : epoch {} - val_acc: {:.6f}".format(best_epoch, best_val_acc))

	if save_best_model:
		model.load_weights(weights_path)

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
