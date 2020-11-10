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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn import svm

def get_content(filename):

    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def get_list_of_reviews(content):
    review_list = []
    for line in content:
        line_list = line.split(" ")
        review_list.append(line_list)
    return review_list

def tranfsform_test_data(unigram_count_dict, test_reviews):

    for review in test_reviews:
        for i in range(len(review)):
            if(unigram_count_dict.get(review[i]) == None):
                review[i] = "<unk>"          
    return test_reviews


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
    all_keys_bi = [key for key, value in bigram_count_dict.items()]
    tokens = 0
    for line in review_list:
        for word in line:
            tokens += 1
    bigram_prob_dict = defaultdict(int)
    for bigram_key in all_keys_bi: #gives probability for BIGRAMS
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
    filename1 = 'DATASET/train/deceptive.txt'
    filename2 = 'DATASET/train/truthful.txt'
    filename3 = 'DATASET/validation/deceptive.txt'
    filename4 = 'DATASET/validation/truthful.txt'
    filename5 = 'DATASET/test/test.txt'

    content1 = get_content(filename1) #training deceptive
    content2 = get_content(filename2) #training truthful
    content3 = get_content(filename3) #validation deceptive
    content4 = get_content(filename4) #validation truthful
    content5 = get_content(filename5) #test

    deceptive_reviews = get_list_of_reviews(content1) #a list ot deceptive reviews
    truthful_reviews = get_list_of_reviews(content2) #a list ot truthful reviews
    test_reviews = get_list_of_reviews(content5) #the list of test reviews

    dec_valid_reviews = get_list_of_reviews(content3)
    tru_valid_reviews = get_list_of_reviews(content4)

    unigram_count_dict_truthful = get_unigram_counts(truthful_reviews)
    unigram_count_dict_deceptive = get_unigram_counts(deceptive_reviews)

    #transformed_test_reviews_truthful = tranfsform_test_data(unigram_count_dict_truthful, test_reviews)
    #transformed_test_reviews_deceptive = tranfsform_test_data(unigram_count_dict_deceptive, test_reviews)

    transformed_valid_tru_reviews_truthful = tranfsform_test_data(unigram_count_dict_truthful, dec_valid_reviews)
    transformed_valid_tru_reviews_deceptive = tranfsform_test_data(unigram_count_dict_deceptive, dec_valid_reviews)

    bigram_count_dict_truthful = get_bigram_counts(truthful_reviews)
    bigram_count_dict_deceptive = get_bigram_counts(deceptive_reviews)

    unigram_prob_dict_truthful = get_unigram_probs(unigram_count_dict_truthful, truthful_reviews)
    unigram_prob_dict_deceptive = get_unigram_probs(unigram_count_dict_deceptive, deceptive_reviews)

    bigram_prob_dict_truthful = get_bigram_probs(bigram_count_dict_truthful, unigram_count_dict_truthful ,truthful_reviews)
    bigram_prob_dict_deceptive = get_bigram_probs(bigram_count_dict_deceptive, unigram_count_dict_deceptive, deceptive_reviews)

    compute_perplex_unigram_truthful = compute_perplex_unigram(transformed_valid_tru_reviews_truthful, unigram_prob_dict_truthful)
    compute_perplex_unigram_deceptive = compute_perplex_unigram(transformed_valid_tru_reviews_deceptive, unigram_prob_dict_deceptive)

    compute_perplex_bigram_truthful = compute_perplex_bigram(transformed_valid_tru_reviews_truthful, unigram_prob_dict_truthful, bigram_prob_dict_truthful)
    compute_perplex_bigram_deceptive = compute_perplex_bigram(transformed_valid_tru_reviews_deceptive, unigram_prob_dict_deceptive, bigram_prob_dict_deceptive)

    unigram_predictions = make_predictions(compute_perplex_unigram_deceptive, compute_perplex_unigram_truthful)
    bigram_predictions = make_predictions(compute_perplex_bigram_deceptive, compute_perplex_bigram_truthful)
    print(f"Unigram Predictions: {unigram_predictions}")
    print(f"Bigram Predictions: {bigram_predictions}")

    dec_train_feats = (get_features_all_reviews(deceptive_reviews, unigram_prob_dict_truthful, unigram_prob_dict_deceptive))
    tru_train_feats = (get_features_all_reviews(truthful_reviews, unigram_prob_dict_truthful, unigram_prob_dict_deceptive))
    label_tru_np = np.zeros(len(truthful_reviews))
    label_dec_np = np.ones(len(deceptive_reviews))
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



 
