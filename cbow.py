import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pickle

import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 200

import codecs
import os
import json

with open('MODEL_CONFIG.json') as f:
    MODEL_CONFIG = json.load(f)

doc = codecs.open('really_small_wiki_sample.txt', 'r', 'utf-8')
content = doc.read()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
corpus = tokenizer.tokenize(content)

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

# used to save the word->integer mapping
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A) #wtf
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

from string import punctuation

remove_terms = punctuation + '0123456789'
corpus = [sent.split(' ') for sent in corpus]

norm_corpus = [[word.lower() for word in sent if word not in remove_terms] for sent in corpus]
norm_corpus = [' '.join(tok_sent) for tok_sent in norm_corpus]
norm_corpus = filter(None, normalize_corpus(norm_corpus))
norm_corpus = [tok_sent for tok_sent in norm_corpus if len(tok_sent.split()) > 2]

print('Total lines:', len(corpus))
print('\nSample line:', corpus[10])
print('\nProcessed line:', norm_corpus[10])

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_corpus)
word2id = tokenizer.word_index

word2id['PAD'] = 0
save_obj(word2id, 'cbow_dict')
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_corpus]

vocab_size = len(word2id)

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])

def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
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
            yield(x, y)

i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=MODEL_CONFIG['window_size'], vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
        if i == 10:
            break
        i += 1

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
from keras.models import model_from_json
import tensorflow as tf

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
from tensorflow.python.client import device_lib

# Run on GPU support
os.environ["CUDA_VISIBLE_DEVICES"]="2"
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


import tensorflow as tf

with tf.Session() as sess:
    print (sess.run(c))

# load json and create model
if(os.path.isfile('-'.join(str(x) for x in MODEL_CONFIG.values()))):
    json_file = open('-'.join(str(x) for x in MODEL_CONFIG.values()), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    cbow = model_from_json(loaded_model_json)
    # load weights into new model
    cbow.load_weights("cbow_weights.h5")
    print("Loaded model from disk")
else:
    cbow = Sequential()
    cbow.add(Embedding(input_dim=vocab_size, output_dim=MODEL_CONFIG['number_of_dimensions_in_hidden_layer'], input_length=MODEL_CONFIG['window_size']*2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(MODEL_CONFIG['number_of_dimensions_in_hidden_layer'],)))
    cbow.add(Dense(vocab_size, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(cbow.summary())

    for epoch in range(1, 10):
        loss = 0.
        i = 0
        for x, y in generate_context_word_pairs(corpus=wids, window_size=MODEL_CONFIG['window_size'], vocab_size=vocab_size):
            i += 1
            loss += cbow.train_on_batch(x, y)
            if i % 100 == 0:
                print('Processed {} (context, word) pairs'.format(i))

        print('Epoch:', epoch, '\tLoss:', loss)
        print()

    model_json = cbow.to_json()
    with open('-'.join(str(x) for x in MODEL_CONFIG.values()), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cbow.save_weights("cbow_weights.h5")
    print("Saved model and weights to disk")

weights = cbow.get_weights()[0]
weights = weights[1:]
print(weights.shape)

