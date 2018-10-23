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


def predictions_bar_plot(pred, true_emotion):
	"""Shows a bar plot of the output of the network for each emotion.
    """
	plt.subplot(1, 2, 2)
	plt.title('Predictions values')
	xticks = range(len(pred))
	color = ['C0'] * 7
	color[np.argmax(pred)] = 'C3'
	color[true_emotion] = 'C2'
	plt.bar(xticks, pred, color=color)
	plt.xticks(xticks, emotions, rotation=45)
	plt.ylim((0, 1))
	plt.show()


def predict_emotion_from_frame(model, emotion, take):
	"""Predicts the emotion of a given frame sequence and display the result.
    """
	sequence = []

	frames = sorted([i for i in os.listdir(frames_path + "/" + emotion + "/" + take) if i.endswith('.jpg')])
	for frame in frames:
		img = imread(frames_path + "/" + emotion + "/" + take + "/" + frame)
		sequence.append(img)
	sequence = np.asarray(sequence)

	vgg_features = extract_vgg_features(sequence[np.newaxis, :])
	ref_img = sequence[nb_frames // 2]
	predict_emotion(model, vgg_features, emotion, ref_img)


def predict_emotion(model, features, true_emotion, ref_img):
	"""Predicts the emotion from the given input features and display the result.
    """
	pred = model.predict(features)[0]
	predicted_emo = emotions[model.predict_classes(features)[0]]

	print("True emotion      : " + true_emotion)
	print("Predicted emotion : " + predicted_emo)

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.imshow(ref_img)
	predictions_bar_plot(pred, emotions.index(true_emotion))


def predict_random(model, features):
	"""Predicts the emotion of a random sample.
    """
	rand_idx = np.random.randint(features.shape[0])
	feat = features[np.newaxis, rand_idx]
	true_emotion = emotions[np.argmax(y[rand_idx])]
	ref_img = x[rand_idx, nb_frames // 2]

	predict_emotion(model, feat, true_emotion, ref_img)