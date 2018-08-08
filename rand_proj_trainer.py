import numpy as np
from sklearn.random_projection import SparseRandomProjection
from keras.utils import np_utils

class rand_proj_trainer:
    def train(self, model_config, pairs, vocab_size, word2id):
        # sp = SparseRandomProjection(n_components=model_config['number_of_dimensions_in_hidden_layer'])
        # projs = []
        # for key, value in word2id.items():
        #     projs.append(np_utils.to_categorical(value, vocab_size))
        # projs = np.array(projs)
        # print('shape before projection:', projs.shape)
        # embedding = sp.fit_transform(projs)
        # print('shape after projection:', embedding.shape)
        # return 0, embedding
        fake_embedding = np.random.normal(0, 0.1, (vocab_size, model_config['number_of_dimensions_in_hidden_layer']))
        return 0, fake_embedding