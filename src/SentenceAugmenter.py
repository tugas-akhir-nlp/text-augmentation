# from __future__ import unicode_literals
import numpy as np
import pandas as pd
from src.srilm import LM
from src.augmentation import BasicTextAugmentation

class SentenceAugmenter:
	def __init__(self, model_path):
		self.lm = LM(model_path.encode())
		self.ta = BasicTextAugmentation()

	def augment_sentence(self, raw_words, augmentation_rate, skip_stopword=True):
		augmented_words = self.ta.generate_augment(raw_words, augmentation_rate, skip_stopword)

		df = pd.DataFrame(columns=['sentence', 'prev1', 'prev2', 'score'])
		for next_word in augmented_words:
			# print(next_word)
			next_df = pd.DataFrame(columns=['sentence', 'token', 'prev1', 'prev2', 'score'])
			if type(next_word) is str:
				# print('processing string')
				if df.size==0:
					print
					score = abs(self.lm.logprob_strings(next_word.encode(), list([])))
					next_df = next_df.append({'sentence': next_word, 'token': next_word, 'prev1': np.nan, 'prev2': np.nan, 'score': score}, ignore_index=True)
					
				else:
					for index, row in df.iterrows():
						sentence = row['sentence']+' '+next_word
						prev2 = row['prev1']
						prev1 = row['token']
						prev_score = row['score']
						if pd.isna(prev1):
							# print('score from 1 word')
							score = abs(self.lm.logprob_strings(next_word.encode(), list([])))
						elif pd.isna(prev2):
							# print('score from 2 word')
							score = abs(self.lm.logprob_strings(next_word.encode(), list([prev1.encode()])))
						else:
							# print('score from 3 word')
							# print(next_word, prev1, prev2)
							score = abs(self.lm.logprob_strings(next_word.encode(), list([prev1.encode(), prev2.encode()])))
						score = score*prev_score
						next_df = next_df.append({'sentence': sentence, 'token': next_word, 'prev1': prev1, 'prev2': prev2, 'score': score}, ignore_index=True)

				df = next_df
				print(df)
				
			else: #list
				# print('processing list')
				if df.size!=0:
					for word in next_word:
						for index, row in df.iterrows():
							sentence = row['sentence']+' '+word
							prev2 = row['prev1']
							prev1 = row['token']
							prev_score = row['score']
							if pd.isna(prev1):
								score = abs(self.lm.logprob_strings(word.encode(), list([])))
							elif pd.isna(prev2):
								score = abs(self.lm.logprob_strings(word.encode(), list([prev1.encode()])))
							else:
								score = abs(self.lm.logprob_strings(word.encode(), list([prev1.encode(), prev2.encode()])))
							score = score*prev_score
							next_df = next_df.append({'sentence': sentence, 'token': word, 'prev1': prev1, 'prev2': prev2, 'score': score}, ignore_index=True)
			
				else:
					for word in next_word:
						sentence = word
						prev2 = np.nan
						prev1 = np.nan
						score = abs(self.lm.logprob_strings(word.encode(), list([])))
						next_df = next_df.append({'sentence': sentence, 'token': word, 'prev1': prev1, 'prev2': prev2, 'score': score}, ignore_index=True)

				next_df.sort_values(by='score', inplace=True)
				df = next_df.head(10)
				print(df)

		df.sort_values(by='score', inplace=True)
		return df.iloc[0]['sentence']

if __name__=='__main__':
	path = 'model/orde3_smoothing.lm'
	augmentation_rate = 0.3
	raw_words = ['nyebelin','banget','bank','bni','tellernya','berbicara','dengan','kasar']

	sa = SentenceAugmenter(path)
	print(sa.augment_sentence(raw_words, augmentation_rate))
