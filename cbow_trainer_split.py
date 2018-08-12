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

class cbow_trainer_split:
    def train(self, model_config, wids, word2id):
        vocab_size = len(word2id.values())
        if model_config['isGPU']:
            # Run on GPU support
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)

        input_context = Input(shape=(model_config['window_size'] * 2,), dtype='int32')
        embedding_one = Embedding(input_dim=vocab_size, output_dim=int(model_config['number_of_dimensions_in_hidden_layer']/2),
                                  input_length=int(model_config['window_size'] * 2), name='embedding_one')(input_context)
        mean = Lambda(lambda x: K.mean(x, axis=1), output_shape=(int(model_config['number_of_dimensions_in_hidden_layer']/2),))
        mean_one = mean(embedding_one)
        predictions = Dense(vocab_size, activation='softmax')(mean_one)
        cbow = Model(inputs=input_context, outputs=predictions)
        cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print('list of layers in cbow:')
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

        layers = [l for l in cbow.layers]
        for i in range(len(layers)):
            layers[i].trainable = False
        embedding_two = Embedding(input_dim=vocab_size, output_dim=int(model_config['number_of_dimensions_in_hidden_layer'] / 2),
                                   input_length=int(model_config['window_size'] * 2), name='embedding_two')(layers[0].output)
        mean_two = mean(embedding_two)
        avg = keras.layers.Concatenate()([layers[2].get_output_at(0), mean_two])
        predictions = Dense(vocab_size, activation='softmax')(avg)
        cbow_after = Model(inputs=layers[0].input, outputs=predictions)
        cbow_after.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print('list of layers in cbow_after:')
        print(cbow_after.summary())

        for epoch in range(1, 10):
            loss = 0.
            i = 0
            for x, y in generate_context_word_pairs(corpus=wids, window_size=model_config['window_size'],
                                                    vocab_size=vocab_size):
                i += 1
                loss += cbow_after.train_on_batch(x, y)
                if i % 100 == 0:
                    print('Processed {} (context, word) pairs'.format(i))

            print('Epoch:', epoch, '\tLoss:', loss)
            print()

        return cbow_after, cbow_after.get_weights()[0]