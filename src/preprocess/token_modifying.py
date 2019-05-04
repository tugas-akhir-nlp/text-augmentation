from utils import list_from_file

vocab = list_from_file('resource/vocabulary.txt')

files_changed = []
files_changed.append('data/corpus_masked_train.txt')

for filename in files_changed:
	data = list_from_file(filename)
	for i in range(len(data)):
		sentence = data[i]
		sequence = sentence.split()
		for j in range(len(sequence)):
			if sequence[j] not in vocab:
				sequence[j] = '<UNK>'
		sentence = ' '.join(sequence)+ '\n'
		data[i] = sentence
	output_filename = filename.split('.')[0]
	output_filename += '_preprocessed.txt'
	with open(output_filename, 'w') as file:
		file.writelines(data)
