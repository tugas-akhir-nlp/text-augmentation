from src.data_generator import Generator

generator = Generator()

for j in range(1, 1001):
	filename = 'data/batch/bengio/6/{}.txt'.format(j)
	with open(filename) as file:
		true_dataset = file.readlines()
	for i in range(len(true_dataset)):
		true_dataset[i] = true_dataset[i].strip().split(';')
	false_dataset = generator.generate_false(true_dataset)
	for i in range(len(false_dataset)):
		false_dataset[i] = ';'.join(false_dataset[i])+ '\n'

	output_filename = 'data/batch/cnw/6_vocab/{}.txt'.format(j)
	print(output_filename)
	with open(output_filename, mode='w') as file:
		file.writelines(false_dataset)
