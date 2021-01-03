import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
import math
import dict2xml
from dicttoxml import dicttoxml
import zlib
from xml_to_dict import XMLtoDict
import ast

class GetData(object):
	def __init__(self):
		self.wordArray= []
		self.walk_dir = "../../../lip_reading/data"
		self.datasets = ["train", "test","val"]
		self.frame_dict =  {"Distance":{},"Angle":{}}
		self.parser = XMLtoDict()
		self.class_count = 3
		self.x_train = np.zeros(shape=(1000*self.class_count ,29,38))
		self.y_train = np.zeros(shape=(1000*self.class_count ))
		self.x_test = np.zeros(shape=(50*self.class_count ,29,38))
		self.y_test = np.zeros(shape=(50*self.class_count ))
		self.x_val= np.zeros(shape=(50*self.class_count ,29,38))
		self.y_val = np.zeros(shape=(50*self.class_count ))
		
	def sortedWalk(self, top, topdown=True, onerror=None):
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
				for x in self.sortedWalk(path, topdown, onerror):
					yield x
		if not topdown:
			yield top, dirs, nondirs

	def getFolderNamesInRootDir(self):
		

		print('walk_dir = ' + self.walk_dir)

		# If your current working directory may change during script execution, it's recommended to
		# immediately convert program arguments to an absolute path. Then the variable root below will
		# be an absolute path as well. Example:
		# walk_dir = os.path.abspath(walk_dir)
		print('walk_dir (absolute) = ' + os.path.abspath(self.walk_dir))

		for root, subdirs, files in self.sortedWalk(self.walk_dir):
			print('--\nroot = ' + root)
			for subdir in sorted(subdirs):
				print('\t- subdirectory ' + subdir)
				self.wordArray.append(subdir)
			break

	def processCompressedFiles(self):
		
		val = np.zeros(shape=(29 ,38))
		k_train = 0
		k_test = 0
		k_val = 0
		print('walk_dir = ' + self.walk_dir)
		for item in self.wordArray:
			if item == 'ACCESS':
				return

			for subitem in self.datasets :
				sourceDir = self.walk_dir +"/" +item + "/" +subitem
				self.targetDir = "data" +"/" +item + "/" +subitem
				for root, subdirs, files in self.sortedWalk(os.path.abspath(sourceDir)):
						
						for file in files:
							if file.endswith(".hgk"):
								filepath = os.path.join(root, file)
								print("processing : ", filepath)
								str_object = zlib.decompress(open(filepath, 'rb').read())

								my_dict = self.parser.parse(str_object)
								i = 0
								for keys in my_dict['test'].keys():
									
									if len(my_dict['test'][keys]['Distance'].values()) != 19 :
										print(keys)
										print(len(my_dict['test'][keys]['Distance'].values()))
										print(len(my_dict['test'][keys]['Angle'].values()))
									temp = np.zeros(shape=(29 ,2 ,19))
									temp[i][0] = np.array(list(map(float, my_dict['test'][keys]['Distance'].values())))
									temp[i][1]= np.array(list(map(float, my_dict['test'][keys]['Angle'].values())))
									val[i] = np.ndarray.flatten(temp[i],'F')
									i = i+1
								if subitem == 'test':
									#self.x_test[k]= np.ndarray.flatten(val)
									self.x_test[k_test]= val
									self.y_test[k_test]= self.wordArray.index(item)+1
									k_test= k_test+1
								elif subitem == 'train':
									#self.x_train[k]= np.ndarray.flatten(val)
									self.x_train[k_train]= val
									self.y_train[k_train]= self.wordArray.index(item)+1
									k_train = k_train+1	
								elif subitem == 'val':
									#self.x_val[k]= np.ndarray.flatten(val)
									self.x_val[k_val]= val
									self.y_val[k_val]= self.wordArray.index(item)+1
									k_val = k_val+1
												
								temp = []




def get_xy_kfolds( word_count=[1], timesteps=1000):
	gd = GetData()
	gd.getFolderNamesInRootDir()
	gd.processCompressedFiles()
	"""
	load exchange rate dataset and preprecess it, then split it into k-folds for CV
	:param split_index: list, the ratio of whole dataset as train set
	:param timesteps: length of a single train x sample
	:return: list, [train_x_set,train_y_set,test_x_single,test_y_single]
	"""
	'''
	df = np.loadtxt('exchange_rate.txt', delimiter=',')
	n = len(df)
	'''
	folds = []
	enc = MinMaxScaler()
	#self.x_train = enc.fit_transform(self.x_train)
	
	#df = enc.fit_transform(df)
	'''
	for split_point in word_count:
		train_end = int(split_point * n)
		train_x, train_y = [], []
		for i in range(train_end - timesteps):
			print(i)
			train_x.append(df[i:i + timesteps])
			train_y.append(df[i + timesteps])
		print(len(train_x))
		print(len(train_y))
		train_x = np.array(train_x)
		train_y = np.array(train_y)
		test_x = df[train_end - timesteps + 1:train_end + 1]
		test_y = df[train_end + 1]
		print(len(test_x))
		print(len(test_y))
		folds.append((train_x, train_y, test_x, test_y))
	'''
	np.save('x_train', gd.x_train)
	np.save('y_train', np.array(gd.y_train))
	np.save('x_val', gd.x_val)
	np.save('y_val', np.array(gd.y_val))
	np.save('x_test', gd.x_test)
	np.save('y_test', np.array(gd.y_test))
	folds.append((np.array(gd.x_train), np.array(gd.y_train), np.array(gd.x_test), np.array(gd.y_test)))
	return folds, enc


if __name__ == '__main__':
	

	get_xy_kfolds()
