with open('resource/vocabulary.txt') as file:
	vocab = file.readlines()
for j in range(len(vocab)):
	vocab[j] = vocab[j].strip()

special_token = ['<S>', '<NUM>']

for j in range(1, 1001):
	filename = 'data/batch/bengio/6/{}.txt'.format(j)
	with open(filename) as file:
		true_dataset = file.readlines()
	for i in reversed(range(len(true_dataset))):
		true_dataset[i] = true_dataset[i].strip().split(';')
		if true_dataset[i][-1] not in vocab:
			true_dataset = true_dataset[:i] + true_dataset[i+1:]
		else:
			for k in range(len(true_dataset[i])-1):
				if true_dataset[i][k] not in vocab:
					if true_dataset[i][k] not in special_token:
						true_dataset[i][k] = '<UNK>'
	for i in range(len(true_dataset)):
		true_dataset[i] = ';'.join(true_dataset[i])+ '\n'
	print(filename)
	with open(filename, 'w') as file:
		file.writelines(true_dataset)
