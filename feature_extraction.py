import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
import re

def read_list_from_file(filename):
    with open(filename) as file:
        temp = file.readlines()
        
    for i in range(len(temp)):
        temp[i] = temp[i].strip()
        
    return temp

class FeatureExtractor:
    def __init__(self,
                 max_length
                ):
        self.max_length = max_length
        self.unk_token = {}
        np.random.seed(9)
        self.unk_token['<s>'] = np.zeros(500)
        self.unk_token['<unk>'] = np.random.rand(500)
        self.unk_token['<num>'] = np.random.rand(500)
        
    def set_w2v(self, w2v_pathname, embedding_len, keep_alive=False):
        self.embedding_len = embedding_len
        self.w2v_pathname = w2v_pathname
        self.keep_w2v = keep_alive
        if self.keep_w2v:
            self.w2v = Word2Vec.load(read_list_from_file(self.w2v_pathname)[0])
        
    def embedding(self, token):
        with open(self.w2v_pathname) as file:
            w2v_path = file.readlines()[0].strip()

        if not self.keep_w2v:
            self.w2v = Word2Vec.load(w2v_path)

        t = Tokenizer(lower=True)
        t.fit_on_texts(token)
        embedding_dict = dict()
        for word in t.word_index:
            word = re.sub('-', '', word)
            word = word.lower()
#             print('finding word vector for:', word)
            if word in self.w2v.wv.vocab:
                embedding_dict[word] = self.w2v.wv[word]
            else:
                if word in self.unk_token:
                    embedding_dict[word] = self.unk_token[word]
                else:
                    print(word, 'is not in word2vec. Generating random vector')
                    embedding_dict[word] = np.random.rand(500)
                    self.unk_token[word] = embedding_dict[word]

        if not self.keep_w2v:
            del self.w2v
        
        embedding = list()
        embedding_len = 500

        for data in token:
            embedding.append([])
            if len(data)>self.max_length:
                data = data[:self.max_length]
            for kata in data:
                kata = re.sub('-', '', kata)
                kata = kata.lower()
                if kata in embedding_dict:
                    embedding[-1].append(embedding_dict[kata])
                else:
                    print(kata, 'not in vocabulary')
                    embedding[-1].append(np.random.rand(embedding_len))
            for i in range(len(embedding[-1]), self.max_length):
                embedding[-1].append(np.zeros(embedding_len))

        return np.array(embedding)