import random
from gensim.models import Word2Vec

with open('../data/corpus.txt') as file:
    corpus = file.readlines()
with open('../w2v_path.txt') as file:
    word2vec_path = file.readlines()[0].strip()
thesaurus_path = '../resource/thesaurus.dat'

true_dataset = []
for i in range(len(corpus)):
# for i in range(1):
    temp = corpus[i].strip().split()
    for j in range(len(temp)-4):
        true_dataset.append(temp[j:j+5])

with open('../resource/dict_id_lower.txt') as file:
    dict_id = file.readlines()
for i in range(len(dict_id)):
    dict_id[i] = dict_id[i].strip('\n')

w2v = Word2Vec.load(word2vec_path)

# Load Thesaurus
thesaurus_file = open(thesaurus_path, 'r')
thesaurus_list = []
thesaurus_map = {}
stopwords_list = []

for idx, line in enumerate(thesaurus_file):
    words = [word.strip() for word in line.split(';')]
    thesaurus_list.append(words)
    for word in words:
        if word not in thesaurus_map:
            thesaurus_map[word] = []
        thesaurus_map[word].append(idx)

def generate_false_dataset(true_dataset):
    false_dataset = []
    for data in true_dataset:
        word = data[-1]
        temp = list(data)
        print('searching subtitute for', word)
    #     data[-1] = random.choice(dict_id)
        if word in thesaurus_map:
            print('terdapat pada thesaurus')
            sub_candidates = set()
            for idx in thesaurus_map[word]:
                for sub_candidate in thesaurus_list[idx]:
                    sub_candidates.add(sub_candidate)

            nearest_words = set()
            real_candidates = set()


            if word in w2v.wv.vocab:
                i = 0
                while len(real_candidates)==0:
                    i += 1
                    print('Percobaan ke', i)
                    for word_tuple in w2v.wv.most_similar(word, topn=i*5):
                        nearest_words.add(word_tuple[0])
                    real_candidates = nearest_words - sub_candidates

                temp[-1] = random.choice(list(real_candidates))
            else:
                temp[-1] = random.choice(list(set(dict_id) - sub_candidates))
        else:
            print('tidak terdapat dalam thesaurus')
            temp[-1] = random.choice(dict_id)
        print()
        false_dataset.append(temp)
        
    return false_dataset

print('Percobaan generasi data')
false_dataset = generate_false_dataset(true_dataset[:10])