import json
from src.preprocessing import number_masking

if __name__=='__main__':
	filenames = []
    filenames.append('../targeted-absa/aspect/data/aspect_train.json')
    filenames.append('../targeted-absa/aspect/data/aspect_test.json')

    for filename in filenames:
	    with open(filename, 'r') as file:
	        aspect_data = json.load(file)

	    for datum in aspect_data:
	        datum['text'] = number_masking(datum['text'])

	    with open(filename[:-5]+'_masked.json', 'w') as file:
	        json.dump(aspect_data, file)
