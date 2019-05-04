import random
from gensim.models import Word2Vec

with open('data/corpus.txt') as file:
    corpus = file.readlines()
with open('resource/w2v_path.txt') as file:
    word2vec_path = file.readlines()[0].strip()
thesaurus_path = 'resource/thesaurus.dat'

with open('resource/vocabulary.txt') as file:
    dict_id = file.readlines()
for i in range(len(dict_id)):
    dict_id[i] = dict_id[i].strip('\n')

w2v = Word2Vec.load(word2vec_path)

class Generator():
    def __init__(self):
        thesaurus_file = open(thesaurus_path, 'r')
        self.thesaurus_list = []
        self.thesaurus_map = {}
        self.subtitution_mapping = {}
        self.unk_word = []

        for idx, line in enumerate(thesaurus_file):
            words = [word.strip() for word in line.split(';')]
            self.thesaurus_list.append(words)
            for word in words:
                if word not in self.thesaurus_map:
                    self.thesaurus_map[word] = []
                self.thesaurus_map[word].append(idx)

    def generate_true(self, corpus):
        true_dataset = []
        for i in range(len(corpus)):
        # for i in range(1):
            temp = corpus[i].strip().split()
            for j in range(len(temp)-4):
                true_dataset.append(temp[j:j+5])
        return true_dataset

    def generate_false(self, true_dataset):
        false_dataset = []
        for data in true_dataset:
            word = data[-1]
            temp = list(data)
            # print('searching subtitute for', word)
            if word not in self.unk_word and word not in self.subtitution_mapping:
                if word in self.thesaurus_map:
                    # terdapat pada thesaurus
                    sub_candidates = set()
                    for idx in self.thesaurus_map[word]:
                        for sub_candidate in self.thesaurus_list[idx]:
                            sub_candidates.add(sub_candidate)

                    nearest_words = set()
                    real_candidates = set()

                    if word in w2v.wv.vocab:
                        i = 0
                        while len(real_candidates)==0:
                            i += 1
                            for word_tuple in w2v.wv.most_similar(word, topn=i*5):
                                nearest_words.add(word_tuple[0])
                            real_candidates = nearest_words - sub_candidates

                        self.subtitution_mapping[word] = list(real_candidates)
                        # temp[-1] = random.choice(list(real_candidates))
                    else:
                        self.subtitution_mapping[word] = list(set(dict_id) - sub_candidates)
                        # temp[-1] = random.choice(list(set(dict_id) - sub_candidates))
                else:
                    # tidak terdapat dalam thesaurus
                    self.unk_word.append(word)
                    # temp[-1] = random.choice(dict_id)

            if word not in self.unk_word:
                temp[-1] = random.choice(self.subtitution_mapping[word])
            else:
                temp[-1] = random.choice(dict_id)
            false_dataset.append(temp)
            
        return false_dataset

def generate_false_dataset(true_dataset):
    false_dataset = []
    for data in true_dataset:
        word = data[-1]
        temp = list(data)
        # print('searching subtitute for', word)
    #     data[-1] = random.choice(dict_id)
        if word in thesaurus_map:
            # print('terdapat pada thesaurus')
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
                    # print('Percobaan ke', i)
                    for word_tuple in w2v.wv.most_similar(word, topn=i*5):
                        nearest_words.add(word_tuple[0])
                    real_candidates = nearest_words - sub_candidates

                temp[-1] = random.choice(list(real_candidates))
            else:
                temp[-1] = random.choice(list(set(dict_id) - sub_candidates))
        else:
            # print('tidak terdapat dalam thesaurus')
            temp[-1] = random.choice(dict_id)
        print()
        false_dataset.append(temp)
        
    return false_dataset

if __name__=='__main__':
    print('Percobaan generasi data')
    false_dataset = generate_false_dataset(true_dataset[:10])