from collections import defaultdict
from collections import Counter
import re
import time
import math
import numpy as np
from feature_extraction import *
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import csv
import pickle
from sklearn import svm

def get_passages_from_unlabeled_txt(filename):
    with open(filename) as f:
        list_of_passages = f.readlines()
    list_of_passages = [passage.strip() for passage in list_of_passages] # strip to remove newline chars
    return list_of_passages

def get_passages_from_labeled_txt(filename):
    list_of_truthful_passages = []
    list_of_deceptive_passages = []
    with open(filename) as f:
        lines = f.readlines()
        for line_idx in range(len(lines)-1):
            if lines[line_idx+1] == "0\n":
                list_of_truthful_passages.append(lines[line_idx])
            elif lines[line_idx+1] == "1\n":
                list_of_deceptive_passages.append(lines[line_idx])
            else:
                pass # line_idx is at a label, not a passage
    list_of_truthful_passages = [passage.strip() for passage in list_of_truthful_passages] # strip to remove newline chars
    list_of_deceptive_passages = [passage.strip() for passage in list_of_deceptive_passages]
    return list_of_truthful_passages, list_of_deceptive_passages


def convert_passages_to_word_lists(passage_list):
    list_of_word_lists = []
    for passage in passage_list:
        word_list = passage.split(" ")
        list_of_word_lists.append(word_list)
    return list_of_word_lists

def insert_unks(passage_list, unigram_count_dict):
    """
    Iterates though all of the words in all of the passages and checks if the word
    is present in unigram_count_dict. If the word is not present, <unk>
    gets placed into its position in the passage.
    :passage_list: list of strings where each string is a passage
    :unigram_count_dict:
    :returns passage_list: the list of input passages with its words not present in unigram_count_dict replaced with unk.
    """
    for passage in passage_list:
        for word_idx in range(len(passage)):
            if(unigram_count_dict.get(passage[word_idx]) == None):
                passage[word_idx] = "<unk>"          
    return passage_list


def get_unigram_counts(list_of_reviews):
    unigram_count_dict = defaultdict(int)
    words_seen_once = {}
    for line in list_of_reviews:
        for word in line:
            if word in words_seen_once.keys():
                unigram_count_dict[word] += 1
            else: #case where this is the first occurence in the word
                unigram_count_dict["<unk>"] += 1 #add 1 to unkown count
                words_seen_once[word] = 0 #add word to words seen once dictionary
    return unigram_count_dict

def get_bigram_counts(list_of_reviews):
    bigram_count_dict = defaultdict(int)
    words_seen_once = {}
    for sentence in list_of_reviews:
        for i in range(len(sentence)-1):
            if ((sentence[i] in words_seen_once.keys()) and (sentence[i+1] in words_seen_once.keys())):
                bigram_count_dict[sentence[i] + " " + sentence[i+1]] += 1
            elif ((sentence[i] in words_seen_once.keys()) and (not sentence[i+1] in words_seen_once.keys())):
                bigram_count_dict[sentence[i] + " " + "<unk>"] += 1
            else: #case where this is the first occurence of both words
                bigram_count_dict["<unk>" + " " + "<unk>"] += 1 #add 1 to unkown count
                words_seen_once[sentence[i]] = 0 #add word to words seen once dictionary
    return bigram_count_dict

def get_unigram_probs(unigram_count_dict, review_list):
    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1
    den = float(tokens) + len(unigram_count_dict.keys())
    unigram_prob_dict  = {k: ((v+1) / den) for k, v in unigram_count_dict.items()}
    return unigram_prob_dict


def get_bigram_probs(bigram_count_dict, unigram_count_dict, review_list):
    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1
    bigram_prob_dict = defaultdict(int)
    for bigram_key in bigram_count_dict.keys():
        bigram_key_count = bigram_count_dict.get(bigram_key)
        first_word = bigram_key.split(" ")[0]
        first_word_count = unigram_count_dict.get(first_word)
        if (first_word_count is None):
            first_word_count = unigram_count_dict.get('<unk>')
        numer2 = float(bigram_key_count) + 1
        den2 = float(first_word_count) + len(unigram_count_dict.keys())
        cond_prob = numer2 / den2   
        bigram_prob_dict[bigram_key] = cond_prob
    return bigram_prob_dict

def compute_perplex_unigram(review_list, unigram_prob_dict):
    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1
    perplex_scores = []
    for line in review_list:
        sum_probs = 0
        for word in line:
            prob = unigram_prob_dict.get(word)
            if prob == None:
                prob = unigram_prob_dict.get("<unk>")
            sum_probs -= math.log(prob)

        result_for_line = math.exp(sum_probs/tokens)
        perplex_scores.append(result_for_line)

    return perplex_scores


def compute_perplex_bigram(review_list, unigram_prob_dict, bigram_prob_dict):
    perplex_scores = []
    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1
    for line in review_list:
        sum_probs = 0

        for i in range(len(line)-1):
            prob = 0
            bigram = str(line[i] + " " + line[i+1])
            if (bigram in bigram_prob_dict.keys()):
                prob = bigram_prob_dict.get(bigram)
            else:
                prob = bigram_prob_dict.get("<unk> <unk>")
            sum_probs -= math.log(prob)
   
        result_for_line = math.exp(sum_probs/tokens)
        perplex_scores.append(result_for_line)

    return perplex_scores


def make_predictions(complexity_deceptive_list, complexity_truthful_list):

    preds = []
    for i in range(len(complexity_deceptive_list)):
        if complexity_deceptive_list[i] < complexity_truthful_list[i]:
            preds.append(1)
        else:
            preds.append(0)
    return preds

def eval_preditions(prediction_list, list_actuals):
    if list_actuals=="truthful":
        true_val = 0
    elif list_actuals=="deceptive":
        true_val = 1
        print(prediction_list)
    correct_count = 0
    for prediction in prediction_list:
        if prediction == true_val:
            correct_count+=1
    accuracy = correct_count/len(prediction_list)
    return accuracy


def get_features_all_reviews(review_list, unigram_prob_dict_tru, unigram_prob_dict_dec):

    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1

    all_features_numpy = np.asarray([])
    for rev in tqdm(review_list):
        rev = " ".join(rev)
        feat_list = get_feature_vector(rev, unigram_prob_dict_tru, unigram_prob_dict_dec, tokens)
        feat_list_numpy = np.asarray(feat_list)
        if all_features_numpy.size == 0:
            all_features_numpy = feat_list_numpy
        else:
            all_features_numpy = np.vstack([all_features_numpy, feat_list_numpy])

    all_features_numpy = normalize(all_features_numpy, axis=0, norm='max')
    return all_features_numpy


if __name__ == "__main__":
    # Load Data
    train_txt_path = 'DATASET/train/train_labeled.txt'
    validation_txt_path = 'DATASET/validation/validation_labeled.txt'

    truthful_train_passages, deceptive_train_passages = get_passages_from_labeled_txt(train_txt_path)
    truthful_validation_passages, deceptive_validation_passages = get_passages_from_labeled_txt(validation_txt_path)

    # Data Preprocessing
    truthful_train_word_lists = convert_passages_to_word_lists(truthful_train_passages)
    deceptive_train_word_lists = convert_passages_to_word_lists(deceptive_train_passages)
    truthful_validation_word_lists = convert_passages_to_word_lists(truthful_validation_passages)
    deceptive_validation_word_lists = convert_passages_to_word_lists(deceptive_validation_passages)

    # Language Model Building
    unigram_count_dict_truthful = get_unigram_counts(truthful_train_word_lists)
    unigram_count_dict_deceptive = get_unigram_counts(deceptive_train_word_lists)

    bigram_count_dict_truthful = get_bigram_counts(truthful_train_word_lists)
    bigram_count_dict_deceptive = get_bigram_counts(deceptive_train_word_lists)

    unigram_prob_dict_truthful = get_unigram_probs(unigram_count_dict_truthful, truthful_train_word_lists)
    unigram_prob_dict_deceptive = get_unigram_probs(unigram_count_dict_deceptive, deceptive_train_word_lists)

    bigram_prob_dict_truthful = get_bigram_probs(bigram_count_dict_truthful, unigram_count_dict_truthful, truthful_train_word_lists)
    bigram_prob_dict_deceptive = get_bigram_probs(bigram_count_dict_deceptive, unigram_count_dict_deceptive, deceptive_train_word_lists)

    # Last bit of data pre-processing
    transformed_valid_reviews_truthful = insert_unks(truthful_validation_word_lists, unigram_count_dict_truthful)
    transformed_valid_reviews_deceptive = insert_unks(deceptive_validation_word_lists, unigram_count_dict_deceptive)

    # Predict
    perplex_unigram_truthful_tprobs = compute_perplex_unigram(transformed_valid_reviews_truthful, unigram_prob_dict_truthful)
    perplex_unigram_truthful_dprobs = compute_perplex_unigram(transformed_valid_reviews_truthful, unigram_prob_dict_deceptive)
    perplex_unigram_deceptive_tprobs = compute_perplex_unigram(transformed_valid_reviews_deceptive, unigram_prob_dict_truthful)
    perplex_unigram_deceptive_dprobs = compute_perplex_unigram(transformed_valid_reviews_deceptive, unigram_prob_dict_deceptive)
    perplex_bigram_truthful_tprobs = compute_perplex_bigram(transformed_valid_reviews_truthful, unigram_prob_dict_truthful, bigram_prob_dict_truthful)
    perplex_bigram_truthful_dprobs = compute_perplex_bigram(transformed_valid_reviews_truthful, unigram_prob_dict_deceptive, bigram_prob_dict_deceptive)
    perplex_bigram_deceptive_tprobs = compute_perplex_bigram(transformed_valid_reviews_deceptive, unigram_prob_dict_truthful, bigram_prob_dict_truthful)
    perplex_bigram_deceptive_dprobs = compute_perplex_bigram(transformed_valid_reviews_deceptive, unigram_prob_dict_deceptive, bigram_prob_dict_deceptive)

    unigram_predictions_truthful = make_predictions(perplex_unigram_truthful_dprobs, perplex_unigram_truthful_tprobs)
    unigram_predictions_deceptive = make_predictions(perplex_unigram_deceptive_dprobs, perplex_unigram_deceptive_tprobs)
    bigram_predictions_truthful = make_predictions(perplex_unigram_truthful_dprobs, perplex_unigram_truthful_tprobs)
    bigram_predictions_deceptive = make_predictions(perplex_unigram_deceptive_dprobs, perplex_unigram_deceptive_tprobs)

    unigram_accuracy_truthful = eval_preditions(unigram_predictions_truthful, "truthful")
    unigram_accuracy_deceptive = eval_preditions(unigram_predictions_deceptive, "deceptive")
    bigram_accuracy_truthful = eval_preditions(bigram_predictions_truthful, "truthful")
    bigram_accuracy_deceptive = eval_preditions(bigram_predictions_deceptive, "deceptive")
    print(f"Unigram accuracy truthful: {unigram_accuracy_truthful}")
    print(f"Unigram accuracy deceptive: {unigram_accuracy_deceptive}")
    print(f"Bigram accuracy truthful: {bigram_accuracy_truthful}")
    print(f"Bigram accuracy deceptive: {bigram_accuracy_deceptive}")

    dec_train_feats = (get_features_all_reviews(deceptive_train_word_lists, unigram_prob_dict_truthful, unigram_prob_dict_deceptive))
    tru_train_feats = (get_features_all_reviews(truthful_train_word_lists, unigram_prob_dict_truthful, unigram_prob_dict_deceptive))
    label_tru_np = np.zeros(len(truthful_train_word_lists))
    label_dec_np = np.ones(len(deceptive_train_word_lists))
    all_feats_train = np.vstack((dec_train_feats, tru_train_feats))
    all_labels_train = np.concatenate((label_dec_np, label_tru_np))

    test_feats = get_features_all_reviews(test_reviews, unigram_prob_dict_truthful, unigram_prob_dict_deceptive)
    valid_feats_dec = get_features_all_reviews(dec_valid_reviews, unigram_prob_dict_truthful, unigram_prob_dict_deceptive)
    valid_feats_tru = get_features_all_reviews(tru_valid_reviews, unigram_prob_dict_truthful, unigram_prob_dict_deceptive)

    model = MultinomialNB()
    model.fit(all_feats_train, all_labels_train)
    preds_test_NB_dec = model.predict(valid_feats_dec)
    preds_test_NB_tru = model.predict(valid_feats_tru)
    preds_test = model.predict(test_feats)



 
