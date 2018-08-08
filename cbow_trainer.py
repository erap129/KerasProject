import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.models import model_from_json
import tensorflow as tf
import os
import keras

class cbow_trainer:
    def train(self, model_config, pairs, vocab_size):
        if model_config['isGPU']:
            # Run on GPU support
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)

        cbow = Sequential()
        cbow.add(Embedding(input_dim=vocab_size, output_dim=model_config['number_of_dimensions_in_hidden_layer'], input_length=model_config['window_size']*2))
        cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(model_config['number_of_dimensions_in_hidden_layer'],)))
        cbow.add(Dense(vocab_size, activation='softmax'))
        cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        print(cbow.summary())

        for epoch in range(1, 10):
            loss = 0.
            i = 0
            for x, y in pairs:
                i += 1
                loss += cbow.train_on_batch(x, y)
                if i % 100 == 0:
                    print('Processed {} (context, word) pairs'.format(i))

            print('Epoch:', epoch, '\tLoss:', loss)
            print()

        return cbow


        # weights = cbow.get_weights()[0]
        # weights = weights[1:]
        # print(weights.shape)