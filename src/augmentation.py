import random
import timeit

class BasicTextAugmentation:
    def __init__(self, thesaurus_path='../resource/thesaurus.dat', stopwords_path='../resource/stopwords.dat', seed=10):
        # Load Thesaurus
        thesaurus_file = open(thesaurus_path, 'r')
        self.thesaurus_list = []
        self.thesaurus_map = {}
        self.stopwords_list = []

        for idx, line in enumerate(thesaurus_file):
            words = [word.strip() for word in line.split(';')]
            self.thesaurus_list.append(words)
            for word in words:
                if word not in self.thesaurus_map:
                    self.thesaurus_map[word] = []
                self.thesaurus_map[word].append(idx)

        # Load Stopword
        stopwords_file = open(stopwords_path, 'r')
        for idx, line in enumerate(stopwords_file):
            self.stopwords_list.append(line.strip())

    def generate_augment(self, words, augmentation_rate=0.3, skip_stopword=True):
        result = []
        widx = []
        tidx = []
        for idx, word in enumerate(words):
            if skip_stopword and word in self.stopwords_list:
                continue

            if word in self.thesaurus_map :
                widx.append(idx)
                tidx.append(self.thesaurus_map[word])

        augmentation_val = augmentation_rate*len(words)
        num_words = int(augmentation_val)
        numbers = [0, 1]
        one_prob = augmentation_val - num_words
        probs = [1-one_prob, one_prob]
        
        num_words += random.choices(numbers, probs)[0]
        
        c_words = words.copy()
        if len(widx) <= num_words:
            print(widx)
            for i in range(len(widx)):
                word_idx = widx[i]
                synonym_list = set()
                for j in range(len(tidx[i])):
                    candidate_synonym = self.thesaurus_list[tidx[i][j]]
                    for synonym in candidate_synonym:
                    	synonym_list.add(synonym)

                c_words[word_idx] = list(synonym_list)
        else:
            ridx = random.sample(range(0, len(widx)-1), num_words)
            print(ridx)
            for i in range(len(ridx)):
                word_idx = widx[ridx[i]]
                synonym_list = set()
                for j in range(len(tidx[ridx[i]])):
                    candidate_synonym = self.thesaurus_list[tidx[ridx[i]][j]]
                    for synonym in candidate_synonym:
                    	synonym_list.add(synonym)

                c_words[word_idx] = list(synonym_list)

        return c_words