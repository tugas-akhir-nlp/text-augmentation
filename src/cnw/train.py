from keras.layers import Flatten, Dense, TimeDistributed, Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from keras.models import Model, load_model, Sequential
from keras import backend as K
import tensorflow as tf
import numpy as np

from feature_extraction import FeatureExtractor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

w2v_pathname = 'resource/w2v_path.txt'

def basic_model():
    n_word = 6
    word_vector_dim = 500
    hidden_unit = 100

    model = Sequential()
    model.add(TimeDistributed(Flatten(), input_shape=(2, n_word, word_vector_dim)))
    model.add(TimeDistributed(Dense(hidden_unit, activation='tanh')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='categorical_hinge',
                  optimizer = 'adagrad',
                  metrics = ['accuracy'])
    
    return model

def cnn_model():
    n_word = 6
    word_vector_dim = 500
    nb_filters = 500
    filter_size_a = 2
    filter_size_b = 3
    drop_rate = 0.5
    
    my_input = Input(shape=(2, n_word, word_vector_dim))

    embedding_dropped = Dropout(drop_rate)(my_input)

    # A branch
    conv_a = TimeDistributed(Conv1D(filters = nb_filters,
                    kernel_size = filter_size_a,
                    activation = 'relu',
                   ))(embedding_dropped)

    pooled_conv_a = TimeDistributed(GlobalMaxPooling1D())(conv_a)

    pooled_conv_dropped_a = Dropout(drop_rate)(pooled_conv_a)

    # B branch
    conv_b = TimeDistributed(Conv1D(filters = nb_filters,
                    kernel_size = filter_size_b,
                    activation = 'relu',
                   ))(embedding_dropped)

    pooled_conv_b = TimeDistributed(GlobalMaxPooling1D())(conv_b)

    pooled_conv_dropped_b = Dropout(drop_rate)(pooled_conv_b)

    #concat
    concat = Concatenate()([pooled_conv_dropped_a, pooled_conv_dropped_b])
    output = TimeDistributed(Dense(units = 1,
                 activation = 'sigmoid',
                 ))(concat)

    model = Model(my_input, output)
    
    model.compile(loss='categorical_hinge',
                  optimizer = 'adagrad',
                  metrics = ['accuracy'])
    
    return model


# print(model.summary())
fe = FeatureExtractor(6)
fe.set_w2v(w2v_pathname, 500, keep_alive=True)

for epoch in range(2, 6):
	model = load_model('model/cnw100_{}.h5'.format(epoch-1))
	for i in range(1,1001):
	    true_filename = 'data/batch/bengio/6/{}.txt'.format(i)
	    false_filename = 'data/batch/cnw/6/{}.txt'.format(i)

	    with open(true_filename) as file:
	        true_data = file.readlines()
	    with open(false_filename) as file:
	        false_data = file.readlines()

	    true_word_seq = []
	    false_word_seq = []
	    label = []
	    for j in range(len(true_data)):
	        true_data[j] = true_data[j].strip()
	        false_data[j] = false_data[j].strip()
	        
	        true_word_seq.append(true_data[j].split(';'))
	        false_word_seq.append(false_data[j].split(';'))
	        label.append([[1], [0]])
	        
	    # print(true_word_seq[0])
	    # print(false_word_seq[0])

	    x_true = fe.embedding(true_word_seq)
	    x_false = fe.embedding(false_word_seq)
	    
	    x = []
	    for j in range(len(x_true)):
	        x.append([x_true[j], x_false[j]])
	    x = np.array(x)
	    
	    y = np.array(label)
	    # print('X shape', x.shape)
	    # print('y shape', y.shape)
	    model.fit(
	        x=x,
	        y=y,
	        batch_size=32,
	        epochs=1)
	    model.save('model/cnw100_{}.h5'.format(epoch))
