import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Lambda, Input, Add
from keras.utils import np_utils
from keras.preprocessing import sequence
import tensorflow as tf
import os
import keras


def generate_context_word_pairs(corpus, vocab_size, window_size=2):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1
            context_words.append([words[i]
                                  for i in range(start, end)
                                  if 0 <= i < sentence_length
                                  and i != index])
            label_word.append(word)
            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)

class cbow_trainer:
    def train(self, model_config, wids, word2id):
        vocab_size = len(word2id.values())
        if model_config['isGPU']:
            # Run on GPU support
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)
        if model_config['split_embedding_layer']:
            # input_context = Input(shape=(model_config['window_size'] * 2, ))
            input_context = Input(shape=(4,), dtype='int32')
            embedding_one = Embedding(input_dim=vocab_size, output_dim=int(model_config['number_of_dimensions_in_hidden_layer']/2),
                                      input_length=int(model_config['window_size'] * 2), name='embedding_one')(input_context)
            embedding_two = Embedding(input_dim=vocab_size, output_dim=int(model_config['number_of_dimensions_in_hidden_layer'] / 2),
                                      input_length=int(model_config['window_size'] * 2), name='embedding_two')(input_context)
            embedding = keras.layers.concatenate([embedding_one, embedding_two])
            mean = Lambda(lambda x: K.mean(x, axis=1), output_shape=(model_config['number_of_dimensions_in_hidden_layer'],))(embedding)
            predictions = Dense(vocab_size, activation='softmax')(mean)
            cbow = Model(inputs=input_context, outputs=predictions)
            cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            print('list of layers in cbow:')
            for layer in cbow.layers:
                print(layer, layer.trainable)
        else:
            cbow = Sequential()
            cbow.add(Embedding(input_dim=vocab_size, output_dim=model_config['number_of_dimensions_in_hidden_layer'], input_length=model_config['window_size']*2))
            cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(model_config['number_of_dimensions_in_hidden_layer'],)))
            cbow.add(Dense(vocab_size, activation='softmax'))
            cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        print(cbow.summary())

        for epoch in range(1, 10):
            loss = 0.
            i = 0
            for x, y in generate_context_word_pairs(corpus=wids, window_size=model_config['window_size'],
                                                    vocab_size=vocab_size):
                i += 1
                loss += cbow.train_on_batch(x, y)
                if i % 100 == 0:
                    print('Processed {} (context, word) pairs'.format(i))

            print('Epoch:', epoch, '\tLoss:', loss)
            print()

        return cbow, cbow.get_weights()[0]