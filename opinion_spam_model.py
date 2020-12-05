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
        self.model_type = config.model_type
    
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
        num_tokens = count_num_of_tokens(review_list)
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
        labeled_train_passages = get_passages_from_labeled_txt(training_data_path)
        train_word_lists = convert_passages_to_word_lists(labeled_train_passages)
        # split training word lists into truthful and deceptive sets
        truthful_train_word_lists = []
        deceptive_train_word_lists = []
        for passage, label in train_word_lists:
            if label == 0:
                truthful_train_word_lists.append(passage)
            else:
                deceptive_train_word_lists.append(passage)
            unigram_count_dict_truthful = get_unigram_counts(truthful_train_word_lists)
        if self.model_type == "unigram":
            unigram_count_dict_truthful = get_unigram_counts(truthful_train_word_lists)
            unigram_count_dict_deceptive = get_unigram_counts(deceptive_train_word_lists)
            unigram_prob_dict_truthful = get_unigram_probs(
                unigram_count_dict_truthful, truthful_train_word_lists
            )
            unigram_prob_dict_deceptive = get_unigram_probs(
                unigram_count_dict_deceptive, deceptive_train_word_lists
            )
            model = {"truthful": unigram_prob_dict_truthful, "deceptive": unigram_prob_dict_deceptive}
        elif self.model_type == "bigram":
            bigram_count_dict_truthful = get_bigram_counts(truthful_train_word_lists, unigram_count_dict_truthful)
            bigram_count_dict_deceptive = get_bigram_counts(deceptive_train_word_lists, unigram_count_dict_deceptive)

            bigram_prob_dict_truthful = get_bigram_probs(
                bigram_count_dict_truthful,
                unigram_count_dict_truthful,
                truthful_train_word_lists,
            )
            bigram_prob_dict_deceptive = get_bigram_probs(
                bigram_count_dict_deceptive,
                unigram_prob_dict_deceptive,
                deceptive_train_word_lists,
            )
            model = {"truthful": bigram_prob_dict_truthful, "deceptive": bigram_prob_dict_deceptive}
        elif self.model_type in ["NB", "Naive Bayes"]:
            pass
        return model
        
        


