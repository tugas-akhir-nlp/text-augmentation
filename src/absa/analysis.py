import json
from keras.preprocessing.text import text_to_word_sequence

with open('data/absa/srilmAugmented02_aspect_train.json') as file:
	data = json.load(file)

count = {}
for i in range(int(len(data)/2)):
	raw_seq = text_to_word_sequence(data[i*2]['sentence'])
	seq = data[i*2+1]['sentence'].split(' ')
	for j in range(len(raw_seq)):
		if raw_seq[j]!=seq[j]:
			key_string = raw_seq[j] + '|' + seq[j]
			if key_string not in count:
				count[key_string] = 0
			count[key_string] += 1

# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_count = sorted(count.items(), key=lambda kv: kv[1])
print(json.dumps(sorted_count, indent=4))
