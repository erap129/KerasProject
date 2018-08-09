import json
import importlib
import datetime
import time
from keras.models import model_from_json

class Blackboard(object):
    def __init__(self):
        self.preprocessor = 0
        self.trainer = 0
        self.evaluator = 0
        self.signer = 0
        self.preprocess_time = 0
        self.train_time = 0
        self.evaluate_time = 0

    def run_pipeline(self, model_config):
        self.preprocessor = getattr(importlib.import_module(model_config['preprocessor']),
                                    model_config['preprocessor'])
        self.trainer = getattr(importlib.import_module(model_config['trainer']),
                                    model_config['trainer'])
        self.evaluator = getattr(importlib.import_module(model_config['evaluator']),
                                 model_config['evaluator'])
        self.signer = getattr(importlib.import_module(model_config['signer']),
                                 model_config['signer'])
        start = time.time()
        wids, word2id = self.preprocessor.preprocess(self.preprocessor, model_config)
        end = time.time()
        self.preprocess_time = end-start
        if model_config['load_weights']:
            json_file = open(model_config['load_model_filename'], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(model_config['load_weights_filename'])
            print("Loaded model from disk")
        else:
            start = time.time()
            model, embedding = self.trainer.train(self.trainer, model_config, wids, word2id)
            end = time.time()
            self.train_time = end-start
        if model_config['save_weights']:
            model_json = model.to_json()
            with open(model_config['save_model_filename'], "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(model_config['save_weights_filename'])
            print("Saved model and weights to disk")
        start = time.time()
        pearson_correlation, pearson_two_tailed_p_value, spearmon_correlation, spearmon_two_tailed_p_value = self.evaluator.evaluate(self.evaluator, model_config, embedding, word2id)
        end = time.time()
        self.evaluate_time = end - start
        correlations = [pearson_correlation, pearson_two_tailed_p_value, spearmon_correlation, spearmon_two_tailed_p_value]
        times = [self.preprocess_time, self.train_time, self.evaluate_time]
        self.signer.sign(self.signer, model_config, correlations, times)


def check_input_size_experiment():
    bb = Blackboard()
    with open('MODEL_CONFIG.json') as f:
        model_config = json.load(f)
        now = datetime.datetime.now()
        sizes = ['14K', '700K', '2.5MB', '5MB']
        model_config['start_time'] = 'input size experiment- ' + now.strftime("%Y-%m-%d %H:%M")
        for configuration_number in range(4):
            model_config['configuration_number'] = configuration_number
            model_config['corpus_filename'] = sizes[configuration_number] + '_sample.txt'
            for i in range(1, 4):
                model_config['configuration_repeat'] = i
                bb.run_pipeline(model_config)

def check_hidden_layer_size_experiment():
    bb = Blackboard()
    with open('MODEL_CONFIG.json') as f:
        model_config = json.load(f)
        now = datetime.datetime.now()
        model_config['start_time'] = 'hidden layer size experiment- ' + now.strftime("%Y-%m-%d %H:%M")
        for configuration_number in range(1, 4):
            model_config['configuration_number'] = configuration_number
            model_config['number_of_dimensions_in_hidden_layer'] = 5 * configuration_number
            for i in range(1, 4):
                model_config['configuration_repeat'] = i
                bb.run_pipeline(model_config)


if __name__ == '__main__':
   check_hidden_layer_size_experiment()
   check_input_size_experiment()






