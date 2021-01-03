from collections import defaultdict
from collections import Counter
import re
import time
import math
import numpy as np
import pandas as pd
from feature_extraction import *
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import csv
import pickle
from sklearn import svm
import yaml

class OpinionSpamModel():
    def __init__(self, config=None, model_type=None):
        self.model_type = config["model_type"]
        self.training_data_path = config["training_data_path"]
        self.validation_data_path = config["validation_data_path"]
    
    @staticmethod
    def get_passages_from_labeled_txt(filename):
        list_of_passages = []
        with open(filename) as f:
            lines = f.readlines()
            for line_idx in range(len(lines) - 1):
                if lines[line_idx + 1] == "0\n":
                    list_of_passages.append((lines[line_idx], 0))
                elif lines[line_idx + 1] == "1\n":
                    list_of_passages.append((lines[line_idx], 1))
                else:
                    pass  # line_idx is at a label, not a passage
        list_of_passages = [
            (passage.strip(), label) for passage, label in list_of_passages
        ]  # strip to remove newline chars
        return list_of_passages
    
    @staticmethod
    def convert_passages_to_word_lists(passage_list):
        list_of_word_lists = []
        for passage, label in passage_list:
            word_list = passage.split(" ")
            list_of_word_lists.append((word_list, label))
        return list_of_word_lists

    @staticmethod
    def count_num_of_tokens(review_list):
        num_tokens = 0
        for line in review_list:
            num_tokens += len(line)
        return num_tokens
    
    @staticmethod
    def get_unigram_counts(list_of_reviews):
        unigram_count_dict = defaultdict(int)
        words_seen_once = {}
        for line in list_of_reviews:
            for word in line:
                unigram_count_dict[word] += 1
        unigram_count_dict["<unk>"] = 1
        return unigram_count_dict
    
    @staticmethod
    def get_bigram_counts(list_of_reviews, unigram_count_dict):
        bigram_count_dict = defaultdict(int)
        words_seen_once = {}
        for sentence in list_of_reviews:
            for i in range(len(sentence) - 1):
                if (sentence[i] in words_seen_once.keys()) and (
                    sentence[i + 1] in words_seen_once.keys()
                ):
                    bigram_count_dict[sentence[i] + " " + sentence[i + 1]] += 1
                elif (sentence[i] in words_seen_once.keys()) and (
                    not sentence[i + 1] in words_seen_once.keys()
                ):
                    bigram_count_dict[sentence[i] + " " + "<unk>"] += 1
                else:  # case where this is the first occurence of both words
                    if (sentence[i + 1] in words_seen_once.keys()):
                        bigram_count_dict["<unk>" + " " + sentence[i]] += 1
                    words_seen_once[
                        sentence[i]
                    ] = 0  # add word to words seen once dictionary
        bigram_count_dict["<unk> <unk>"] = 1  # add 1 to unkown count
        return bigram_count_dict
    
    @staticmethod
    def get_unigram_probs(unigram_count_dict, review_list):
        """
        Converts count dictionary into a probability dictionary.
        """
        num_tokens = OpinionSpamModel.count_num_of_tokens(review_list)
        token_count_smoothed = num_tokens + len(unigram_count_dict.keys())
        unigram_prob_dict = {k: ((v + 1) / token_count_smoothed) for k, v in unigram_count_dict.items()}
        return unigram_prob_dict

    @staticmethod
    def get_bigram_probs(bigram_count_dict, unigram_count_dict, review_list):
        # Using formula: P(Bigram|FirstWord) = P(FirstWord and Bigram) / P(FirstWord)
        word_dict_with_unk = defaultdict(int)
        words_seen_once = {}
        for line in review_list:
            for word in line:
                if word in words_seen_once.keys():
                    word_dict_with_unk[word] += 1
                else: # case where this is the first occurence in the word
                    word_dict_with_unk["<unk>"] += 1  # add 1 to unkown count
                    words_seen_once[word] = 0  # add word to words seen once dictionary
        bigram_prob_dict = defaultdict(int)
        for bigram_key in bigram_count_dict.keys():
            bigram_key_count = bigram_count_dict.get(bigram_key)
            first_word = bigram_key.split(" ")[0]
            first_word_count = word_dict_with_unk.get(first_word)
            if first_word_count is None:
                first_word_count = word_dict_with_unk.get("<unk>")
            count_bigram_smoothed = bigram_key_count + 1
            count_first_word_smoothed = first_word_count + len(bigram_count_dict.keys())
            cond_prob = count_bigram_smoothed / count_first_word_smoothed
            bigram_prob_dict[bigram_key] = cond_prob
        return bigram_prob_dict
    
    def train(self, training_data_path=None):
        if training_data_path is None:
            training_data_path = self.training_data_path
        labeled_train_passages = OpinionSpamModel.get_passages_from_labeled_txt(training_data_path)
        train_word_lists = OpinionSpamModel.convert_passages_to_word_lists(labeled_train_passages)
        # split training word lists into truthful and deceptive sets
        truthful_train_word_lists = []
        deceptive_train_word_lists = []
        for passage, label in train_word_lists:
            if label == 0:
                truthful_train_word_lists.append(passage)
            else:
                deceptive_train_word_lists.append(passage)
        unigram_count_dict_truthful = OpinionSpamModel.get_unigram_counts(truthful_train_word_lists)
        unigram_count_dict_deceptive = OpinionSpamModel.get_unigram_counts(deceptive_train_word_lists)
        unigram_prob_dict_truthful = OpinionSpamModel.get_unigram_probs(
            unigram_count_dict_truthful, truthful_train_word_lists
        )
        unigram_prob_dict_deceptive = OpinionSpamModel.get_unigram_probs(
            unigram_count_dict_deceptive, deceptive_train_word_lists
        )
        model = {"unigram_truthful": unigram_prob_dict_truthful, "unigram_deceptive": unigram_prob_dict_deceptive}
        if self.model_type == "bigram": # add bigram dictionaries to model if model type is bigram
            bigram_count_dict_truthful = OpinionSpamModel.get_bigram_counts(truthful_train_word_lists, unigram_count_dict_truthful)
            bigram_count_dict_deceptive = OpinionSpamModel.get_bigram_counts(deceptive_train_word_lists, unigram_count_dict_deceptive)

            bigram_prob_dict_truthful = OpinionSpamModel.get_bigram_probs(
                bigram_count_dict_truthful,
                unigram_count_dict_truthful,
                truthful_train_word_lists,
            )
            bigram_prob_dict_deceptive = OpinionSpamModel.get_bigram_probs(
                bigram_count_dict_deceptive,
                unigram_prob_dict_deceptive,
                deceptive_train_word_lists,
            )
            model["bigram_truthful"] = bigram_prob_dict_truthful
            model["bigram_deceptive"] = bigram_prob_dict_deceptive
        elif self.model_type in ["NB", "Naive Bayes"]:
            pass
        self.trained_model = model
        return model
    
    @staticmethod
    def insert_unks(passage_list, unigram_count_dict_truthful, unigram_count_dict_deceptive):
        """
        Iterates though all of the words in all of the passages and checks if the word
        is present in unigram_count_dict. If the word is not present, <unk>
        gets placed into its position in the passage.
        :passage_list: list of strings where each string is a passage
        Returns the list of input passages, where the words not present in unigram_count_dict are replaced with unk.
        """
        for passage, label in passage_list:
            if label == 0:
                count_dict = unigram_count_dict_truthful
            else:
                count_dict = unigram_count_dict_deceptive
            for word_idx in range(len(passage)):
                if count_dict.get(passage[word_idx]) is None:
                    passage[word_idx] = "<unk>"
        return passage_list

    def compute_perplex(self, review_list, probability_dict):
        num_tokens = OpinionSpamModel.count_num_of_tokens(review_list)
        perplex_scores = []
        if self.model_type == "unigram":
            for line in review_list:
                sum_probs = 0
                for word in line:
                    prob = probability_dict.get(word)
                    if prob is None:
                        prob = probability_dict.get("<unk>")
                    sum_probs -= math.log(prob)

                result_for_line = math.exp(sum_probs / num_tokens)
                perplex_scores.append(result_for_line)
        elif self.model_type == "bigram":
            for line in review_list:
                sum_probs = 0
                for i in range(len(line) - 1):
                    prob = 0
                    bigram = str(line[i] + " " + line[i + 1])
                    prob = probability_dict.get(bigram)
                    if prob is None:
                        prob = probability_dict.get("<unk> <unk>")
                    sum_probs -= math.log(prob)
                result_for_line = math.exp(sum_probs / num_tokens)
                perplex_scores.append(result_for_line)
        return perplex_scores
    
    @staticmethod
    def eval_preditions(prediction_list, list_actuals):
        if list_actuals == "truthful":
            true_val = 0
        elif list_actuals == "deceptive":
            true_val = 1
        correct_count = 0
        for prediction in prediction_list:
            if prediction == true_val:
                correct_count += 1
        accuracy = correct_count / len(prediction_list)
        return accuracy
        
    @staticmethod
    def make_predictions(complexity_deceptive_list, complexity_truthful_list):
        """
        Return the class with the higher perplexity score.
        """
        preds = []
        for i in range(len(complexity_deceptive_list)):
            if complexity_deceptive_list[i] < complexity_truthful_list[i]:
                preds.append(1)
            else:
                preds.append(0)
        return preds
    
    @staticmethod
    def calculate_average_accuracy(truthful_accuracy, truthful_predictions, deceptive_accuracy, deceptive_predictions):
        average_accuracy = (
            truthful_accuracy * len(truthful_predictions) + 
            deceptive_accuracy * len(deceptive_predictions)
        ) / (len(truthful_predictions) + len(deceptive_predictions))
        return average_accuracy

    
    def predict(self, validation_data_path=None):
        if validation_data_path is None:
            validation_data_path = self.validation_data_path
        unigram_prob_dict_truthful = self.trained_model["unigram_truthful"]
        unigram_prob_dict_deceptive = self.trained_model["unigram_deceptive"]
        labeled_validation_passages = OpinionSpamModel.get_passages_from_labeled_txt(validation_data_path)
        validation_word_lists = OpinionSpamModel.convert_passages_to_word_lists(labeled_validation_passages)
        truthful_validation_word_lists = []
        deceptive_validation_word_lists = []
        for passage, label in validation_word_lists:
            if label == 0:
                truthful_validation_word_lists.append(passage)
            else:
                deceptive_validation_word_lists.append(passage)
        transformed_validation_word_lists = OpinionSpamModel.insert_unks(validation_word_lists, unigram_prob_dict_truthful, unigram_prob_dict_deceptive)
        transformed_valid_reviews_truthful = []
        transformed_valid_reviews_deceptive = []
        for passage, label in transformed_validation_word_lists:
            if label == 0:
                transformed_valid_reviews_truthful.append(passage)
            else:
                transformed_valid_reviews_deceptive.append(passage)
        if self.model_type in ["unigram", "bigram"]:
            perplex_truthful_tprobs = self.compute_perplex(
                transformed_valid_reviews_truthful, self.trained_model[self.model_type + "_truthful"]
            )
            perplex_truthful_dprobs = self.compute_perplex(
                transformed_valid_reviews_truthful, self.trained_model[self.model_type + "_deceptive"]
            )
            perplex_deceptive_tprobs = self.compute_perplex(
                transformed_valid_reviews_deceptive, self.trained_model[self.model_type + "_truthful"]
            )
            perplex_deceptive_dprobs = self.compute_perplex(
                transformed_valid_reviews_deceptive, self.trained_model[self.model_type + "_deceptive"]
            )
            predictions_truthful = OpinionSpamModel.make_predictions(
                perplex_truthful_dprobs, perplex_truthful_tprobs
            )
            predictions_deceptive = OpinionSpamModel.make_predictions(
                perplex_deceptive_dprobs, perplex_deceptive_tprobs
            )
            accuracy_truthful = OpinionSpamModel.eval_preditions(
                predictions_truthful, "truthful"
            )
            accuracy_deceptive = OpinionSpamModel.eval_preditions(
                predictions_deceptive, "deceptive"
            )
            average_accuracy = OpinionSpamModel.calculate_average_accuracy(
                accuracy_truthful,
                predictions_truthful,
                accuracy_deceptive,
                predictions_deceptive,
            )
            prediction_report = {"accuracy": average_accuracy}
            return prediction_report

if __name__ == "__main__":
    with open(r'new_config.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    for model in config.keys():
        OpSpamModel = OpinionSpamModel(config[model])
        OpSpamModel.train()
        prediction_report = OpSpamModel.predict()
        print(prediction_report)


