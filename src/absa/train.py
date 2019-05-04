import json
from src.metrics import f1
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Input, Embedding, concatenate, Bidirectional, LSTM, GlobalMaxPool1D, Dropout, Dense
from utils import list_from_file
from keras import backend as K
import tensorflow as tf
import numpy as np
from feature_extraction import FeatureExtractor
from sklearn.model_selection import train_test_split
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

class AspectModel:
    def __init__(self, input_layers, n_aspect):
        self.input_layers = input_layers
        self.n_aspect = n_aspect
        self.build_model()
        
    def build_model(self):
        first_input = []
        concat_input = []
        
        for layer in self.input_layers:
            if layer['embedding']:
                first_input.append(Input(shape=layer['shape'][1:], name=layer['name']))
                concat_input.append(Embedding(output_dim=layer['embedding_len'], input_dim=layer['embedding_vocab'], input_length=self.max_length)(first_input[-1]))
            else:
                first_input.append(Input(shape=layer['shape'][1:], name=layer['name']))
                concat_input.append(first_input[-1])
        
        # concat = concatenate(concat_input, axis=-1)
        hidden = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(first_input[-1])
        hidden = GlobalMaxPool1D()(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(64, activation='relu')(hidden)
        hidden = Dropout(0.5)(hidden)
        output = Dense(self.n_aspect, activation='sigmoid')(hidden)
        
        self.model = Model(inputs=first_input, outputs=output)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adagrad',
            metrics=[f1]
        )

    def data_split(self, x, y, test_size, random_state=10):
        init_data_size = 600

        random.seed(random_state)
        shuffle_idx = [a for a in range(init_data_size)]
        random.shuffle(shuffle_idx)
        split_idx = int(init_data_size*(1-test_size))

        x_train = x_val = np.empty((0,50,500), float)
        y_train = y_val = np.empty((0,6), int)
        for i in range(split_idx):
            x_train = np.append(x_train, x[shuffle_idx[i]*2:shuffle_idx[i]*2+2], axis=0)
            y_train = np.append(y_train, y[shuffle_idx[i]*2:shuffle_idx[i]*2+2], axis=0)
        for i in range(split_idx, init_data_size):
            x_val = np.append(x_val, x[shuffle_idx[i]*2:shuffle_idx[i]*2+2], axis=0)
            y_val = np.append(y_val, y[shuffle_idx[i]*2:shuffle_idx[i]*2+2], axis=0)

        return x_train, x_val, y_train, y_val

    def train(self, x, y, max_epoch):
        print(len(x))
        print(x.shape)
        if len(x)==600:
            print('Using sklearn split')
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=10)
        else:
            print('Using split strategy which keep single datum and its augmented in one group')
            x_train, x_val, y_train, y_val = self.data_split(x, y, test_size=0.2)
        save_filename = 'model/temp_aspect.h5'
        mc = ModelCheckpoint(save_filename, 
            monitor='val_f1',
            save_best_only=True,
            mode='max')

        self.model.fit(
            x_train, 
            y_train, 
            batch_size=32, 
            epochs=max_epoch, 
            validation_data=[x_val, y_val],
            callbacks=[mc])

        self.model = load_model(save_filename, 
            custom_objects={'f1': f1})

    def evaluate(self, x_test, y_test):
        print(self.model.evaluate(x_test, y_test))

aspect_list = list_from_file('resource/aspect.txt')
n_aspect = len(aspect_list)
w2v_pathname = 'resource/w2v_path.txt'
max_length = 50
fe = FeatureExtractor(max_length)
fe.set_w2v(w2v_pathname, 500, keep_alive=True)

def prepare_feature(filename):
    with open(filename) as file:
        data = json.load(file)

    sentences = [datum['sentence'] for datum in data]
    aspects = [datum['aspect'] for datum in data]

    sequences = [text_to_word_sequence(s) for s in sentences]
    
    label = []
    for i in range(len(aspects)):
        label.append(np.zeros(n_aspect))
        for a in aspects[i]:
            label[-1][aspect_list.index(a)] = 1

    x = fe.embedding(sequences)
    y = np.array(label)

    return x, y

def train():
    train_filename = 'data/absa/'+ input('Train file: ')
    test_filename = 'data/absa/aspect_test.json'
    x, y = prepare_feature(train_filename)
    x_test, y_test = prepare_feature(test_filename)

    input_layers = []
    input_layers.append({
        'shape': x.shape,
        'embedding': False,
        'name': 'word_embedding'
        })

    am = AspectModel(input_layers, n_aspect)
    am.model.summary()
    am.train(x, y, 20)
    am.evaluate(x_test, y_test)
    model_name = train_filename.split('/')[-1]
    model_name = model_name.split('.')[0]+ '.h5'
    am.model.save('model/absa/'+ model_name)

if __name__=='__main__':
    train()
