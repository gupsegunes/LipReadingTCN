from tcn import compiled_tcn
from utils import get_xy_kfolds
from sklearn.metrics import mean_squared_error
import numpy as np

# dataset source: https://github.com/laiguokun/multivariate-time-series-data
# exchange rate: the collection of the daily exchange rates of eight foreign countries
# including Australia, British, Canada, Switzerland, China, Japan, New Zealand and
# Singapore ranging from 1990 to 2016.
# task: predict multi-column daily exchange rate from history

#folds, enc = get_xy_kfolds()
#mse_list = []

if __name__ == '__main__':
	mse_list = []
	print('loading saved data...')
	x_train = np.load('x_train.npy')
	y_train = np.load('y_train.npy')

	x_val = np.load('x_val.npy')
	y_val = np.load('y_val.npy')

	x_test = np.load('x_test.npy')
	y_test = np.load('y_test.npy')
	#for x_train, y_train, x_test, y_test in folds:
	'''
	model = compiled_tcn(return_sequences=False,
					num_feat=x_train.shape[2],
					num_classes=4, 
					nb_filters=20,
					kernel_size=6,
					dilations=[2 ** i for i in range(9)],
					nb_stacks=1,
					max_len=x_train.shape[1],
					use_skip_connections=True)
	'''
	model = compiled_tcn(return_sequences=False,
					num_feat=x_train.shape[2],
					num_classes=4, 
					nb_filters=20,
					kernel_size=6,
					dilations=[2 ** i for i in range(3)],
					nb_stacks=10,
					max_len=x_train.shape[1],
					use_skip_connections=False)
	print(f'x_train.shape = {x_train.shape}')
	print(f'y_train.shape = {y_train.shape}')
	print(f'x_test.shape = {x_test.shape}')
	print(f'y_test.shape = {y_test.shape}')

	model.summary()

	model.fit(x_train, y_train, epochs=100,
		validation_data=(x_test, y_test))
	print(f"final loss on test set: {np.mean(mse_list)}")
