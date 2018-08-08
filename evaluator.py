import pandas as pd
from keras.models import model_from_json
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pickle
import os
import json

__author__ = "Aviad Elyashar"

model_config = 0

# used to load the word->index dictionary
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def remove_pairs_not_exist_in_vocabulary(similarity_results_df, targeted_trained_model_df):
    indexes_to_remove = similarity_results_df[similarity_results_df['cosine_similarity'] == -1].index.tolist()
    similarity_results_df = similarity_results_df.drop(similarity_results_df.index[indexes_to_remove])
    similarity_results_df = similarity_results_df.reset_index()

    targeted_trained_model_df = targeted_trained_model_df.drop(targeted_trained_model_df.index[indexes_to_remove])
    targeted_trained_model_df = targeted_trained_model_df.reset_index()

    return similarity_results_df, targeted_trained_model_df


def calculate_cosine_similarity_on_evaluation_dataset(targeted_trained_model_df, weights, word2id):
    # if os.path.isfile('-'.join(str(x) for x in MODEL_CONFIG.values())):
    #     json_file = open('-'.join(str(x) for x in MODEL_CONFIG.values()), 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     trained_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     trained_model.load_weights("cbow_weights.h5")
    #     weights = trained_model.get_weights()[0]
    #     print("Loaded model from disk")

    # word2index = load_obj(word_index_dict_name)
    # print('word2index:', word2index)

    result_tuples = []
    pairs_exist_in_vocabulary = []
    pairs_not_exist_in_vocabulary = []
    for index, row in targeted_trained_model_df.iterrows():
        word_1 = row['word_1']
        word_2 = row['word_2']
        human_similarity = row['similarity']

        try:
            word_1_vector = weights[word2id[word_1]]
            word_2_vector = weights[word2id[word_2]]
            similarity_result = 1 - spatial.distance.cosine(word_1_vector, word_2_vector)

            result_tuple = (word_1, word_2, similarity_result)

            pairs_exist_in_vocabulary.append(result_tuple)
            result_tuples.append(result_tuple)

        except KeyError as e:
            print(e)
            error_tuple = (word_1, word_2, -1)

            pairs_not_exist_in_vocabulary.append(error_tuple)
            result_tuples.append(error_tuple)

    print("Pairs exist in vocabulary: {0}".format(len(pairs_exist_in_vocabulary)))
    print("Pairs not exist in vocabulary: {0}".format(len(pairs_not_exist_in_vocabulary)))
    print("Total pairs: {0}".format(len(result_tuples)))
    print("---------------------------------------------")

    return result_tuples


def calculate_pearson_correlation(human_similarity_results, cosine_similarity_results):
    pearson_correlation_and_two_tailed_p_value_tuple = pearsonr(human_similarity_results, cosine_similarity_results)

    pearson_correlation = pearson_correlation_and_two_tailed_p_value_tuple[0]
    pearson_two_tailed_p_value = pearson_correlation_and_two_tailed_p_value_tuple[1]
    print("Pearson_correlation: {0}".format(pearson_correlation))
    print("Pearson_two_tailed_p_value: {0}".format(pearson_two_tailed_p_value))
    print("---------------------------------------------")

    return pearson_correlation, pearson_two_tailed_p_value


def calculate_spearmon_correlation(human_similarity_results, cosine_similarity_results):
    spearmon_result = spearmanr(human_similarity_results, cosine_similarity_results)

    spearmon_correlation = spearmon_result.correlation
    spearmon_two_tailed_p_value = spearmon_result.pvalue
    print("Spearmon_correlation: {0}".format(spearmon_correlation))
    print("Spearmon_two_tailed_p_value: {0}".format(spearmon_two_tailed_p_value))
    print("---------------------------------------------")

    return spearmon_correlation, spearmon_two_tailed_p_value


def print_correlation_results(targeted_trained_model_full_path, pearson_correlation, pearson_two_tailed_p_value,
                              spearmon_correlation, spearmon_two_tailed_p_value):
    model_name = targeted_trained_model_full_path.split("/")[-1]
    parameters = model_name.split("-")

    parameters.append(pearson_correlation)
    parameters.append(pearson_two_tailed_p_value)
    parameters.append(spearmon_correlation)
    parameters.append(spearmon_two_tailed_p_value)

    correlation_result_tuples = tuple(parameters)
    print('correlation_result_tuples:', correlation_result_tuples)
    correlation_result_df = pd.DataFrame([correlation_result_tuples], columns=['dataset', 'window_size', 'epochs', 'number_of_dimensions_in_hidden_layer', 'selected_optimizer', 'pearson_correlation', 'pearson_two_tailed_p_value', 'spearmon_correlation', 'spearmon_two_tailed_p_value'])

    # full path for the file of the correlation results
    correlation_result_file_full_path = "output/correlation_results.csv"
    if not os.path.exists(correlation_result_file_full_path):
        correlation_result_df.to_csv(correlation_result_file_full_path, index=None)

    else:
        with open(correlation_result_file_full_path, 'a') as f:
            correlation_result_df.to_csv(f, index=None)


def calculate_correlations_and_print_results(similarity_results_df, targeted_trained_model_df, targeted_trained_model_full_path):
    human_similarity_results = targeted_trained_model_df['similarity']
    cosine_similarity_results = similarity_results_df['cosine_similarity']

    pearson_correlation, pearson_two_tailed_p_value = calculate_pearson_correlation(human_similarity_results, cosine_similarity_results)

    spearmon_correlation, spearmon_two_tailed_p_value = calculate_spearmon_correlation(human_similarity_results, cosine_similarity_results)

    return pearson_correlation, pearson_two_tailed_p_value, spearmon_correlation, spearmon_two_tailed_p_value
    # print_correlation_results(targeted_trained_model_full_path, pearson_correlation, pearson_two_tailed_p_value, spearmon_correlation,
    #                           spearmon_two_tailed_p_value)


class evaluator:
    def evaluate(self, model_config, weights, word2id):
        #full path for the file with the cosine similarity results over the evaluation dataset with the selected trained model
        cosine_result_file_path = "output/cosine_similarity_results_over_target_dataset_with_trained_model.csv"

        # full path file for evaluation dataset
        targeted_evaluation_dataset_full_path = model_config['dataset'] + '.csv'
        targeted_trained_model_df = pd.read_csv(targeted_evaluation_dataset_full_path)

        # full path file for trained model
        targeted_trained_model_full_path = '-'.join(str(x) for x in model_config.values())

        #multipication_coefficient
        multi_coeff = 5
        cosine_similarities_tuples = calculate_cosine_similarity_on_evaluation_dataset(targeted_trained_model_df, weights, word2id)

        similarity_results_df = pd.DataFrame(cosine_similarities_tuples, columns=['word_1', 'word_2', 'cosine_similarity'])

        similarity_results_df['cosine_similarity'] = similarity_results_df['cosine_similarity'].apply(lambda x: x * multi_coeff)

        similarity_results_df.to_csv(cosine_result_file_path, index=None)

        similarity_results_df, targeted_trained_model_df = remove_pairs_not_exist_in_vocabulary(similarity_results_df, targeted_trained_model_df)

        return calculate_correlations_and_print_results(similarity_results_df, targeted_trained_model_df, targeted_trained_model_full_path)

