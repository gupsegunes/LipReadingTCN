import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import cv2
import os

walk_dir = "../../lip_reading/data"
wordArray = ['ABOUT','ABUSE']
datasets = ['train','test','val']

def sortedWalk(top, topdown=True, onerror=None):
	from os.path import join, isdir, islink

	names = os.listdir(top)
	names.sort()
	dirs, nondirs = [], []

	for name in names:
		if isdir(os.path.join(top, name)):
			dirs.append(name)
		else:
			nondirs.append(name)

	if topdown:
		yield top, dirs, nondirs
	for name in dirs:
		path = join(top, name)
		if not os.path.islink(path):
			for x in sortedWalk(path, topdown, onerror):
				yield x
	if not topdown:
		yield top, dirs, nondirs

def data_generator():
	val = np.zeros(shape=(48 ,48))
	k_train = 0
	k_test = 0
	k_val = 0
	'''
	x_train = np.zeros(shape=(2000 ,29))
	y_train= np.zeros(shape=(2000 ,1))
	x_test= np.zeros(shape=(50 ,29))
	y_test = np.zeros(shape=(50 ,29))
	x_val = np.zeros(shape=(50 ,29))
	y_val = np.zeros(shape=(50 ,29))
	'''
	x_train =np.zeros((2000, 29,48,48))
	
	y_train=np.zeros((2000, 1))
	x_test= np.zeros((100, 29,48,48))
	y_test = np.zeros((100, 1))
	x_val =np.zeros((100, 29,48,48))
	y_val = np.zeros((100, 1))
	print('walk_dir = ' + walk_dir)
	temp = np.zeros((29,48,48))
	for item in wordArray:

		#index = 1
		for subitem in datasets :
			sourceDir = walk_dir +"/" +item + "/" +subitem
			targetDir = "data" +"/" +item + "/" +subitem
			index = 1
			for root, subdirs, files in sortedWalk(os.path.abspath(sourceDir)):
					
					for file in files:
						if file.endswith(".jpg"):
							
							filepath = os.path.join(root, file)
							print("processing : ", filepath)

							img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

							if index == int(file[6:-10]):
								temp[int(file[12:-4])]= img
							else:
								
								if subitem == 'test':
									#self.x_test[k]= np.ndarray.flatten(val)
									x_test[index-1]= temp
									y_test[index-1]= wordArray.index(item)
									k_test= k_test+1
								elif subitem == 'train':
									#self.x_train[k]= np.ndarray.flatten(val)
									x_train[index-1]= temp
									y_train[index-1]= wordArray.index(item)
									k_train = k_train+1	
								elif subitem == 'val':
									#self.x_val[k]= np.ndarray.flatten(val)
									x_val[index-1]= temp
									y_val[index-1]= wordArray.index(item)
									k_val = k_val+1
								temp = np.zeros((29,48,48))
								temp[int(file[12:-4])]= img

							index = int(file[6:-10])
							index2 = int(file[12:-4])

			
	# input image dimensions
	img_rows, img_cols = 48, 48
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#x_train = np.array(x_train)
	#x_test = np.array(x_test)
	x_train = x_train.reshape(-1,29*img_rows * img_cols, 1)
	x_test = x_test.reshape(-1,29*img_rows * img_cols, 1)

	num_classes = 2
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)

	y_train = np.expand_dims(y_train, axis=2)
	y_test = np.expand_dims(y_test, axis=2)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	np.save('x_train',x_train)
	np.save('y_train', y_train)
	np.save('x_val', x_val)
	np.save('y_val', y_val)
	np.save('x_test', x_test)
	np.save('y_test', y_test)
	return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
	print(data_generator())
