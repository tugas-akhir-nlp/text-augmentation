from src.bengio.augment import BengioAugmenter
from utils import list_from_file
from math import log10

model_name = input('Model: ')
ba = BengioAugmenter('model/'+model_name)

logprob = 0
n_word = 0
val_data = list_from_file('data/corpus_masked_test_preprocessed.txt')
for sentence in val_data:
	print(sentence)
	seq = sentence.split(' ')
	if len(seq)>=6:
		for i in range(len(seq)-5):
			n_word += 1
			seq_query = seq[i:i+6]
			prob = ba.prob(seq_query)
			logprob += (-1)*log10(prob)

print('Log probability', logprob)
print('Banyak kata', n_word)
perplexity = 10**(logprob/n_word)
print('Perplexity', perplexity)