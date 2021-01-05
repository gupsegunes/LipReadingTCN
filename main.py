from utils import data_generator
import numpy as np
from tcn import compiled_tcn


def run_task():
	print('loading saved data...')
	x_train = np.load('x_train.npy')
	y_train = np.load('y_train.npy')

	x_val = np.load('x_val.npy')
	y_val = np.load('y_val.npy')

	x_test = np.load('x_test.npy')
	y_test = np.load('y_test.npy')

	model = compiled_tcn(return_sequences=False,
							num_feat=1,
							num_classes=10,
							nb_filters=20,
							kernel_size=6,
							dilations=[2 ** i for i in range(9)],
							nb_stacks=1,
							max_len=x_train[0:1].shape[1],
							use_skip_connections=True)

	print(f'x_train.shape = {x_train.shape}')
	print(f'y_train.shape = {y_train.shape}')
	print(f'x_test.shape = {x_test.shape}')
	print(f'y_test.shape = {y_test.shape}')

	model.summary()

	model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
				validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':
	run_task()
