import numpy as np
import pandas as pd
from src.srilm import LM
from src.augmentation import BasicTextAugmentation
from utils import list_from_file
from feature_extraction import FeatureExtractor
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

ta = BasicTextAugmentation()
vocab = list_from_file('resource/vocabulary.txt')
w2v_pathname = 'resource/w2v_path.txt'
fe = FeatureExtractor(5)
fe.set_w2v(w2v_pathname, 500, keep_alive=True)

class BengioAugmenter():
	def __init__(self, model_filename):
		self.gram = 5
		self.model = load_model(model_filename)

	def prob(self, sequences):
		for i in range(len(sequences)):
			if sequences[i] not in vocab:
				if sequences[i].isdigit():
					sequences[i] = '<NUM>'
				else:
					sequences[i] = '<UNK>'
		feature = sequences[:-1]
		while len(feature)<self.gram:
			feature = ['<S>']+ feature
		# print('Feature', feature)
		x = fe.embedding([feature])
		y_pred = self.model.predict(x)[0]

		word_label = sequences[-1]
		# print('Label', word_label)
		label_idx = vocab.index(word_label)
		chance = y_pred[label_idx]

		return chance

	def augment_sentence(self, raw_words, augmentation_rate, skip_stopword=True):
		augmented_words = ta.generate_augment(raw_words, augmentation_rate, skip_stopword)

		df = pd.DataFrame(columns=['sentence', 'score'])
		for next_word in augmented_words:
			next_df = pd.DataFrame(columns=['sentence', 'score'])
			if type(next_word) is str:
				if df.size==0:
					score = abs(self.prob([next_word]))
					next_df = next_df.append({'sentence': next_word, 'score': score}, ignore_index=True)
				else:
					for index, row in df.iterrows():
						sentence = row['sentence']+' '+next_word
						seq = sentence.split()
						if len(seq)>self.gram:
							seq = seq[self.gram*(-1):]
						score = self.prob(seq)
						prev_score = row['score']
						score = score*prev_score
						next_df = next_df.append({'sentence': sentence, 'score': score}, ignore_index=True)
				df = next_df
				print(df)
			else: #list
				if df.size!=0:
					for word in next_word:
						for index, row in df.iterrows():
							sentence = row['sentence']+' '+word
							prev_score = row['score']
							seq = sentence.split()
							if len(seq)>self.gram:
								seq = seq[self.gram*(-1):]
							score = self.prob(seq)
							score = score*prev_score
							next_df = next_df.append({'sentence': sentence, 'score': score}, ignore_index=True)
				else:
					for word in next_word:
						sentence = word
						score = self.prob([word])
						next_df = next_df.append({'sentence': sentence, 'score': score}, ignore_index=True)

				next_df.sort_values(by='score', ascending=False, inplace=True)
				df = next_df.head(10)
				print(df)

		df.sort_values(by='score', ascending=False, inplace=True)
		return df.iloc[0]['sentence']


if __name__=='__main__':
	ba = BengioAugmenter('model/bengio_sgd.h5')
	augmentation_rate = 0.3
	
	with open('data/absa/aspect_test.json') as file:
		data = json.load(file)

	temp_data = []
	for datum in data:
		temp_data.append(datum)

		temp_datum = dict()
		temp_datum['sentence'] = ba.augment_sentence(text_to_word_sequence(datum['sentence']), augmentation_rate)
		temp_datum['aspect'] = list(datum['aspect'])

		temp_data.append(temp_datum)

	with open('data/absa/bengioAugmented03_aspect_test.json', 'w') as file:
		json.dump(temp_data, file, indent=4)
