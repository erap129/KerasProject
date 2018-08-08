import pandas as pd
import numpy as np
import re
import pickle
import nltk
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from string import punctuation
nltk.download('punkt')
nltk.download('stopwords')
pd.options.display.max_colwidth = 200

import codecs

# used to save the word->integer mapping
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def normalize_document(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)  # wtf
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc


def generate_context_word_pairs(corpus, window_size, vocab_size):
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


class preprocessor:
    def preprocess(self, model_config):
        doc = codecs.open(model_config['corpus_filename'], 'r', 'utf-8')
        content = doc.read()

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        corpus = tokenizer.tokenize(content)

        normalize_corpus = np.vectorize(normalize_document)
        remove_terms = punctuation + '0123456789'
        corpus = [sent.split(' ') for sent in corpus]

        norm_corpus = [[word.lower() for word in sent if word not in remove_terms] for sent in corpus]
        norm_corpus = [' '.join(tok_sent) for tok_sent in norm_corpus]
        norm_corpus = filter(None, normalize_corpus(norm_corpus))
        norm_corpus = [tok_sent for tok_sent in norm_corpus if len(tok_sent.split()) > 2]

        print('Total lines:', len(corpus))
        print('\nSample line:', corpus[10])
        print('\nProcessed line:', norm_corpus[10])

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

        result_pairs = []
        for x, y in generate_context_word_pairs(corpus=wids, window_size=model_config['window_size'], vocab_size=vocab_size):
            result_pairs.append([x, y])
        # returns the word-window pairs, and the mapping between word and integer as well (for later use by evaluator)
        return result_pairs, word2id, vocab_size

