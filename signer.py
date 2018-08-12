import os


class signer:
    def sign(self, model_config, times, correlations):
        if os.path.isfile("output/results.csv") is False:
            with open("output/results.csv", 'w') as csv:
                header = ''
                for key in model_config.keys():
                    header += key + ','
                # header = header[:-1]
                # csv.write("start time, configuration number, configuration repeat, isGPU, preprocessor, trainer, evaluator, signer, corpus_filename, dataset, window_size, epochs, number_of_dimensions_in_hidden_layer, selected_optimizer, save_weights, save_weights_filename, save_model_filename, load_weights, load_weights_filename, load_model_filename, preprocess time, train time, evaluate time, pearson correlation, pearson 2 tail p-value, spearman correlation, spearman 2 tail p-value\n")
                header += "preprocess time,train time,evaluate time,pearson correlation,pearson 2 tail p-value,spearman correlation,spearman 2 tail p-value\n"
                csv.write(header)
        csv = open("output/results.csv", 'a')
        line = ""
        for value in model_config.values():
            line += str(value) + ","
        for value in correlations:
            line += str(value) + ","
        for value in times:
            line += str(value) + ","
        line = line[:-1]
        line += "\n"
        csv.write(line)
