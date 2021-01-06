from utils import data_generator
import numpy as np
from tcn import compiled_tcn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from utils import wordCount
import os

batch_size = 16 
now = datetime.now()
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

def evaluate_model(model,y_test,y_val):
	print('Evaluating the model...')
	score = model.evaluate(x_val, y_val, batch_size=config.batch_size)
	print('Finished training, with the following val score:')
	print(score)
	print('Evaluating the model...')
	score = model.evaluate(x_test, y_test, batch_size=config.batch_size)
	print('Finished training, with the following val score:')
	print(score)

def create_save_plots(history,model):
	create_plots(history)
	plot_and_save_cm(model)

def plot_and_save_cm(model):
	now = datetime.now()
	fileName = 'plots/conf_matrix_test_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
	y_pred = model.predict_classes(self.test_data, verbose=1)
	self.plot_confusion_matrix(self.y_test, y_pred, classes=self.config.class_names,fileName=fileName)

	fileName = 'plots/conf_matrix_val_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))
	y_pred = model.predict_classes(self.val_data, verbose=1)
	self.plot_confusion_matrix(self.y_val, y_pred, classes=self.config.class_names,fileName=fileName)

def create_plots(history):
	if not os.path.exists('plots'):
		os.mkdir('plots')

	now = datetime.now()
	# summarize history for accuracy
	print("create_plots {0}".format())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')

	fileName = 'plots/acc_plot_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))


	plt.savefig(fileName)
	plt.clf()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')

	fileName = 'plots/loss_plot_{0}.png'.format(now.strftime("%d_%m_%Y_%H%M%S"))


	plt.savefig(fileName)
	plt.clf()


def plot_confusion_matrix(y_true, y_pred, classes,fileName,
						normalize=False,
						title=None,
						cmap=plt.cm.Blues,
						):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	
	plt.savefig(fileName)
	plt.clf()
	return ax

def run_task():
	print('loading saved data...')
	x_train = np.load('x_train.npy')
	y_train = np.load('y_train.npy')

	x_val = np.load('x_val.npy')
	y_val = np.load('y_val.npy')

	x_test = np.load('x_test.npy')
	y_test = np.load('y_test.npy')

	y_train_one_hot_label = to_categorical(y_train, wordCount)
	y_test_one_hot_label = to_categorical(y_test, wordCount)
	y_val_one_hot_label = to_categorical(y_val, wordCount)

	y_train_one_hot_label = np.expand_dims(y_train_one_hot_label, axis=2)
	y_test_one_hot_label = np.expand_dims(y_test_one_hot_label, axis=2)
	y_val_one_hot_label = np.expand_dims(y_val_one_hot_label, axis=2)

	model = compiled_tcn(return_sequences=False,
							num_feat=1,
							num_classes=2,
							nb_filters=20,
							kernel_size=6,
							dilations=[2 ** i for i in range(9)],
							nb_stacks=1,
							max_len=x_train[0:1].shape[1],
							use_skip_connections=True)

	print(f'x_train.shape = {x_train.shape}')
	print(f'y_train.shape = {y_train_one_hot_label.shape}')
	print(f'x_test.shape = {x_test.shape}')
	print(f'y_test.shape = {y_test_one_hot_label.shape}')

	model.summary()
	now = datetime.now()
	plot_file_name = 'tcn_model_plot_' + now.strftime("%d_%m_%Y_%H%M%S") + '.png'
	plot_model(model, to_file=plot_file_name, show_shapes=True, show_layer_names=True)

	history = model.fit(x_train, y_train_one_hot_label.squeeze().argmax(axis=1), epochs=1,
				validation_data=(x_test, y_test_one_hot_label.squeeze().argmax(axis=1)),callbacks=[tensorboard_callback])

	create_save_plots(history,model)
	evaluate_model(model,y_test_one_hot_label,y_val_one_hot_label)



if __name__ == '__main__':
	run_task()
