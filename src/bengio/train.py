from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from numpy import array
from feature_extraction import FeatureExtractor
from keras import backend as K
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
K.tensorflow_backend.set_session(tf.Session(config=config))

w2v_pathname = 'resource/w2v_path.txt'

with open('resource/vocabulary.txt') as file:
	vocab = file.readlines()
	num_class = len(vocab)
for i in range(len(vocab)):
	vocab[i] = vocab[i].strip()

def build_model(n_word):
	global num_class
	model = Sequential()
	model.add(Flatten(input_shape=(n_word, 500)))
	model.add(Dropout(0.1))
	model.add(Dense(num_class, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
		optimizer='sgd')
	return model

# model = load_model('model/bengio_sgd_2.h5')
# print(model.summary())
fe = FeatureExtractor(5)
fe.set_w2v(w2v_pathname, 500, keep_alive=True)

for epoch in range(9, 11):
	model = load_model('model/bengio_sgd_{}.h5'.format(epoch-1))
	for i in range(1,1001):
		filename = 'data/batch/bengio/6/{}.txt'.format(i)
		print(filename)
		with open(filename) as file:
			data = file.readlines()
		word_seq = []
		label = []
		for j in range(len(data)):
			data[j] = data[j].strip()
			splitted = data[j].split(';')
			word_seq.append(splitted[:-1])
			word_label = splitted[-1]
			temp_one_hot = np.zeros(num_class)
			one_idx = vocab.index(word_label)
			if one_idx==-1:
				print(word_label)
			temp_one_hot[one_idx] = 1
			label.append(temp_one_hot)
		x = fe.embedding(word_seq)
		y = np.array(label)
		# print('X shape', x.shape)
		# print('y shape', y.shape)
		model.fit(
			x=x,
			y=y,
			batch_size=32,
			epochs=1)
		model.save('model/bengio_sgd_{}.h5'.format(epoch))
