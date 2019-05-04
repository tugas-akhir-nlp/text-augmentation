from keras.models import Sequential
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from feature_extraction import FeatureExtractor
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

with open('resource/vocabulary.txt') as file:
	vocab = file.readlines()
for j in range(len(vocab)):
	vocab[j] = vocab[j].strip()

special_token = ['<S>', '<NUM>']

for i in range(2,1001):
	filename = 'data/batch/{}.txt'.format(i)
	with open(filename) as file:
		batch_data = file.readlines()
	for j in range(len(batch_data)):
		batch_data[j] = batch_data[j].strip()

	word_seq = {}
	for j in range(6,7):
		word_seq[j] = []
	
	for datum in batch_data:
		splitted = datum.split(' ')
		for j in range(6,7):
			for k in range(len(splitted)):
				begin_idx = k-j+1
				if begin_idx<0:
					begin_idx = 0
				end_idx = k+1
				temp = splitted[begin_idx:end_idx]
				while(len(temp)<j):
					temp = ['<S>'] + temp
				for l in range(len(temp)):
					if temp[l] not in vocab:
						if temp[l] not in special_token:
							temp[l] = '<UNK>'
				word_seq[j].append(';'.join(temp)+ '\n')

	for j in range(6,7):
		output_filename = 'data/batch/bengio/{}+unk/{}.txt'.format(j, i)
		print(output_filename)
		with open(output_filename, mode='w') as file:
			file.writelines(word_seq[j])
