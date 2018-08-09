import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Reshape, merge
from keras.models import Model

class skipgram_trainer:
    def train(self, model_config, wids, word2id):
        vocab_size = len(word2id.values())
        sampling_table = sequence.make_sampling_table(vocab_size)
        wids_flat = [word for sentence in wids for word in sentence]
        couples, labels = skipgrams(wids_flat, vocab_size, window_size=model_config['window_size'], sampling_table=sampling_table)
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")

        input_target = Input((1,))
        input_context = Input((1,))
        vector_dim = model_config['number_of_dimensions_in_hidden_layer']

        embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
        target = embedding(input_target)
        target = Reshape((vector_dim, 1))(target)
        context = embedding(input_context)
        context = Reshape((vector_dim, 1))(context)

        # setup a cosine similarity operation which will be output in a secondary model
        similarity = merge([target, context], mode='cos', dot_axes=0)

        # now perform the dot product operation to get a similarity measure
        dot_product = merge([target, context], mode='dot', dot_axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        model = Model(input=[input_target, input_context], output=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        validation_model = Model(input=[input_target, input_context], output=similarity)

        class SimilarityCallback:
            def run_sim(self):
                valid_size = 16  # Random set of words to evaluate similarity on.
                valid_window = 100  # Only pick dev samples in the head of the distribution.
                valid_examples = np.random.choice(valid_window, valid_size, replace=False)
                reverse_dictionary = dict(zip(word2id.values(), word2id.keys()))
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    sim = self._get_sim(valid_examples[i])
                    nearest = (-sim).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

            @staticmethod
            def _get_sim(valid_word_idx):
                sim = np.zeros((vocab_size,))
                in_arr1 = np.zeros((1,))
                in_arr2 = np.zeros((1,))
                in_arr1[0,] = valid_word_idx
                for i in range(vocab_size):
                    in_arr2[0,] = i
                    out = validation_model.predict_on_batch([in_arr1, in_arr2])
                    sim[i] = out
                return sim

        sim_cb = SimilarityCallback()

        arr_1 = np.zeros((1,))
        arr_2 = np.zeros((1,))
        arr_3 = np.zeros((1,))
        for cnt in range(model_config['epochs']):
            idx = np.random.randint(0, len(labels) - 1)
            arr_1[0,] = word_target[idx]
            arr_2[0,] = word_context[idx]
            arr_3[0,] = labels[idx]
            loss = model.train_on_batch([arr_1, arr_2], arr_3)
            if cnt % 100 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))
            if cnt % 10000 == 0:
                sim_cb.run_sim()

        return model, model.get_weights()[0]

