from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np
import itertools

from const import *

def plot_roc_curve(y_pred, y_true, driver):
	plt.figure()
	lw = 2
	colors = ['darkorange', 'navy', 'aqua', 'green']

	for i in np.unique(y_true)+1:
		print(i)
		fpr, tpr, thresholds = roc_curve(y_true, y_pred[:,i-1], i-1)
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw = 2, color = colors[i-1],label='ROC curve class %d (area = %0.2f)' % (i,roc_auc))
		
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC for driver {}'.format(driver))
	plt.legend(loc="lower right")
	plt.savefig(results_path + driver + "_ROCcurve.png")
	plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, comments=None):

	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	if not comments is None:
		plt.text(-2.6,4.6,comments, fontsize = 12)
	plt.savefig(results_path + title + ".png")
	np.set_printoptions(precision=2)
	plt.close()
