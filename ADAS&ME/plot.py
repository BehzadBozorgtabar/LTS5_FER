import numpy as np
import matplotlib.pyplot as plt

from const import *

def prepare_standardplot(title, xlabel):
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.set_figheight(4)
	fig.set_figwidth(10)
	fig.suptitle(title)
	ax1.set_title('Loss')
	ax1.set_ylabel('categorical cross entropy')
	ax1.set_xlabel(xlabel)
	ax1.set_yscale('log')
	ax2.set_title('Accuracy')
	ax2.set_ylabel('accuracy [% correct]')
	ax2.set_xlabel(xlabel)
	return fig, ax1, ax2


def finalize_standardplot(fig, ax1, ax2):
	ax1handles, ax1labels = ax1.get_legend_handles_labels()
	if len(ax1labels) > 0:
		ax1.legend(ax1handles, ax1labels)
	ax2handles, ax2labels = ax2.get_legend_handles_labels()
	if len(ax2labels) > 0:
		ax2.legend(ax2handles, ax2labels)
	fig.tight_layout()
	plt.subplots_adjust(top=0.9)


def plot_history(history, title):
	fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
	ax1.plot(history.history['loss'], label = "training")
	ax1.plot(history.history['val_loss'], label = "validation")
	ax2.plot(history.history['acc'], label = "training")
	ax2.plot(history.history['val_acc'], label = "validation")
	finalize_standardplot(fig, ax1, ax2)
	plt.show()


def plot_histories(histories, title):
	fig, ax1, ax2 = prepare_standardplot(title, 'epoch')
	alpha = 0.5
	ax1.plot(histories[0].history['loss'], 'C0', label = "training", alpha=alpha)
	ax1.plot(histories[0].history['val_loss'], 'C1', label = "validation", alpha=alpha)
	ax2.plot(histories[0].history['acc'], 'C0', label = "training", alpha=alpha)
	ax2.plot(histories[0].history['val_acc'], 'C1', label = "validation", alpha=alpha)
	
	for history in histories[1:]:
		ax1.plot(history.history['loss'], 'C0', alpha=alpha)
		ax1.plot(history.history['val_loss'], 'C1', alpha=alpha)
		ax2.plot(history.history['acc'], 'C0', alpha=alpha)
		ax2.plot(history.history['val_acc'], 'C1', alpha=alpha)
	
	finalize_standardplot(fig, ax1, ax2)
	plt.show()


def plot_confusion_matrix(cm,
						  target_names=None,
						  title='Confusion matrix',
						  normalize=False):
	
	accuracy = np.trace(cm) / float(np.sum(cm))

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
	plt.title(title)
	plt.colorbar()
	
	if target_names is not None:
		ticks = np.arange(len(target_names))
		plt.xticks(ticks, target_names, rotation=45)
		plt.yticks(ticks, target_names)

	thresh = 0.6*cm.max()
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			if normalize:
				plt.text(j, i, "{:0.4f}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black")
			else:
				plt.text(j, i, "{:,}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\n\nAccuracy={:0.4f}'.format(accuracy))
	plt.show()