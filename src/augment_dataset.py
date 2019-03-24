import json
from src.SentenceAugmenter import SentenceAugmenter
from keras.preprocessing.text import text_to_word_sequence

path = 'model/orde3_smoothing.lm'
augmentation_rate = 0.3

sa = SentenceAugmenter(path)

with open('../targeted-absa/aspect/data/aspect_train_masked.json') as file:
	data = json.load(file)

temp_data = []
for datum in data:
	temp_data.append(datum)

	temp_datum = dict()
	temp_datum['aspects'] = list(datum['aspects'])
	temp_datum['text'] = sa.augment_sentence(text_to_word_sequence(datum['text']), augmentation_rate)

	temp_data.append(temp_datum)

with open('../targeted-absa/aspect/data/augmented_aspect_train.json', 'w') as file:
	json.dump(temp_data, file)
