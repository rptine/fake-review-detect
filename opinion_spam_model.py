from collections import defaultdict
from collections import Counter
import re
import time
import math
import numpy as np
import pandas as pd
import feature_extraction as feat_extract
from tqdm import tqdm
from sklearn import preprocessing, naive_bayes
import csv
import pickle
import yaml

class OpinionSpamModel():
    """
    Opinion Spam Model

    Encapsulation of training and evaluating three types (unigram, bigram and Naive Bayes) of 
    opinion spam models. Opinion spam models predict whether text reviews are truthful or
    deceptive.

    Attributes:
        model_type: one of unigram, bigram or Naive Bayes
        training_data_path: Path to .txt file containing training data
        validation_data_path: Path to .txt file containing validation data
    """

    def __init__(self, config=None, model_type=None, training_data_path=None, validation_data_path=None):
        """
        Initializes OpinionSpamModel with a model type, training_data_path and validation_data_path.

        Args:
            filename: path to .txt file containing a labeled list of reviews, where each review is
            an entire line in the file, and the following line is its label (0 -> truthful, 1 -> 
            deceptive).        
        """
        self.model_type = config["model_type"]
        self.training_data_path = config["training_data_path"]
        self.validation_data_path = config["validation_data_path"]
    
    @staticmethod
    def get_passages_from_labeled_txt(filename):
        """
        Reads text file located at filename, which contains text reviews with labels, removes any 
        leading white space from the reviews, converts reviews from strings to lists of words, and 
        builds a list of (review, label) tuples.

        Args:
            filename: path to .txt file containing a labeled list of reviews, where each review is
            an entire line in the file, and the following line is its label (0 -> truthful, 1 -> 
            deceptive).

        Returns:
            A list of (review, label) tuples where a review is a list of words
        """
        list_of_labeled_passages = []
        with open(filename) as f:
            lines = f.readlines()
            for line_idx in range(len(lines) - 1):
                if lines[line_idx + 1] == "0\n":
                    list_of_labeled_passages.append((lines[line_idx], 0))
                elif lines[line_idx + 1] == "1\n":
                    list_of_labeled_passages.append((lines[line_idx], 1))
                else:
                    pass  # line_idx is at a label, not a passage
        list_of_labeled_passages = [
            (passage.strip().split(" "), label) for passage, label in list_of_labeled_passages
        ]  # strip to remove newline chars
        return list_of_labeled_passages

    @staticmethod
    def count_num_of_tokens(list_of_word_lists):
        """
        Finds the total number of tokens contained in a list of strings.

        Args:
            list_of_word_lists: a list of lists of words, where each word is a string

        Returns:
            The total number of tokens in the list of strings, where a token is a
            non-unique word within the string.
        """
        num_tokens = 0
        for line in list_of_word_lists:
            num_tokens += len(line)
        return num_tokens
    
    @staticmethod
    def build_unigram_count_dict(list_of_word_lists):
        """
        Produces a dictionary of unigram counts, where each unique word in the list of 
        word lists is a key, and the key's corresponding value is the number of occurrences
        of the word.

        Args:
            list_of_word_lists: a list of word lists

        Returns:
            A dictionary of unigram counts, mapping unique words in list_of_word_lists
            to the number of occurrences.
        """
        unigram_count_dict = defaultdict(int)
        for line in list_of_word_lists:
            for word in line:
                unigram_count_dict[word] += 1
        unigram_count_dict["<unk>"] = 1
        return unigram_count_dict
    
    @staticmethod
    def build_bigram_count_dict(list_of_word_lists, unigram_count_dict):
        """
        Produces a dictionary of bigram counts, where each unique pair of successive words in 
        list_of_word_lists is a key, and the key's corresponding value is the number of occurrences
        of the pair of words.

        Args:
            list_of_word_lists: a list of word lists

        Returns:
            A dictionary of bigram counts, mapping unique pairs of successive words in 
            list_of_word_lists to the number of occurrences.
        """
        # unigram_count_dict =
        bigram_count_dict = defaultdict(int)
        words_seen_once = {}
        for sentence in list_of_word_lists:
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
    def get_unigram_prob_dict(unigram_count_dict, list_of_word_lists):
        """
        Produces a dictionary of unigram probabilities for all unigrams in the unigram_count_dict.

        Args:
            unigram_count_dict: a dictionary mapping unigrams (single words) to their respective
            counts
            list_of_word_lists: a list of word lists

        Returns:
            A dictionary mapping unigrams to their respective probabilities for all unigrams in the
            unigram_count_dict. The probability is calculated from relative the counts of each
            unigram and incorporates plus-one-smoothing.
        """
        num_tokens = OpinionSpamModel.count_num_of_tokens(list_of_word_lists)
        token_count_smoothed = num_tokens + len(unigram_count_dict.keys())
        unigram_prob_dict = {k: ((v + 1) / token_count_smoothed) for k, v in unigram_count_dict.items()}
        return unigram_prob_dict

    @staticmethod
    def get_bigram_prob_dict(bigram_count_dict, unigram_count_dict, list_of_word_lists):
        """
        Produces a dictionary of bigram probabilities for all bigrams in the bigram_count_dict.

        Args:
            bigram_count_dict: a dictionary mapping bigrams (pairs of successive words) to their respective counts
            in the list_of_word_lists
            unigram_count_dict: a dictionary mapping unigrams (single words) to their respective
            counts in the list_of_word_lists
            list_of_word_lists: a list of word lists

        Returns:
            A dictionary mapping bigrams to their respective probabilities, for all bigrams in the
            bigram_count_dict. The probability is calculated from relative the counts of each
            bigram_count_dict and incorporates plus-one-smoothing.
        """
        # Using formula: P(Bigram|FirstWord) = P(FirstWord and Bigram) / P(FirstWord)
        word_dict_with_unk = defaultdict(int)
        words_seen_once = {}
        for line in list_of_word_lists:
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
        """
        Encapsulation of logic for training three types (unigram, bigram and Naive Bayes) of 
        opinion spam models.

        Attributes:
            training_data_path: Path to .txt file containing training data

        Returns
            OpinionSpamModel object with the attribute trained_model containing a reference to
            to a model trained on the data obtained from the training_data_path.
        """
        if training_data_path is None:
            training_data_path = self.training_data_path
        labeled_train_word_lists = OpinionSpamModel.get_passages_from_labeled_txt(training_data_path)
        # split training word lists into truthful and deceptive sets
        truthful_train_word_lists = []
        deceptive_train_word_lists = []
        for word_list, label in labeled_train_word_lists:
            if label == 0:
                truthful_train_word_lists.append(word_list)
            else:
                deceptive_train_word_lists.append(word_list)
        unigram_count_dict_truthful = OpinionSpamModel.build_unigram_count_dict(truthful_train_word_lists)
        unigram_count_dict_deceptive = OpinionSpamModel.build_unigram_count_dict(deceptive_train_word_lists)
        unigram_prob_dict_truthful = OpinionSpamModel.get_unigram_prob_dict(
            unigram_count_dict_truthful, truthful_train_word_lists
        )
        unigram_prob_dict_deceptive = OpinionSpamModel.get_unigram_prob_dict(
            unigram_count_dict_deceptive, deceptive_train_word_lists
        )
        self.unigram_truthful = unigram_prob_dict_truthful
        self.unigram_deceptive = unigram_prob_dict_deceptive
        model = {"unigram_truthful": unigram_prob_dict_truthful, "unigram_deceptive": unigram_prob_dict_deceptive}
        if self.model_type == "bigram": # add bigram dictionaries to model if model type is bigram
            bigram_count_dict_truthful = OpinionSpamModel.build_bigram_count_dict(truthful_train_word_lists, unigram_count_dict_truthful)
            bigram_count_dict_deceptive = OpinionSpamModel.build_bigram_count_dict(deceptive_train_word_lists, unigram_count_dict_deceptive)
            bigram_prob_dict_truthful = OpinionSpamModel.get_bigram_prob_dict(
                bigram_count_dict_truthful,
                unigram_prob_dict_truthful,
                truthful_train_word_lists,
            )
            bigram_prob_dict_deceptive = OpinionSpamModel.get_bigram_prob_dict(
                bigram_count_dict_deceptive,
                unigram_prob_dict_deceptive,
                deceptive_train_word_lists,
            )
            model["bigram_truthful"] = bigram_prob_dict_truthful
            model["bigram_deceptive"] = bigram_prob_dict_deceptive
        elif self.model_type in ["NB", "Naive Bayes"]:
            dec_train_feats = OpinionSpamModel.get_features_all_reviews(
                deceptive_train_word_lists,
                unigram_prob_dict_truthful,
                unigram_prob_dict_deceptive,
            )
            tru_train_feats = OpinionSpamModel.get_features_all_reviews(
                truthful_train_word_lists,
                unigram_prob_dict_truthful,
                unigram_prob_dict_deceptive,
            )
            label_tru_np = np.zeros(len(truthful_train_word_lists))
            label_dec_np = np.ones(len(deceptive_train_word_lists))
            all_feats_train = np.vstack((dec_train_feats, tru_train_feats))
            all_labels_train = np.concatenate((label_dec_np, label_tru_np))
            model = naive_bayes.MultinomialNB()
            model.fit(all_feats_train, all_labels_train)
        self.trained_model = model
        return model
    
    @staticmethod
    def insert_unks(list_of_word_lists, words_seen_during_training):
        """
        Replaces strings in the list_of_word_lists that do not appear as keys in 
        words_seen_during_training with the string <unk>.

        Arguments:
            list_of_word_lists: a list of word lists
            words_seen_during_training: a list of words seen during training
        
        Returns: 
            A modified version of the input list_of_word_lists, where each word not present in
            words_seen_during_training is replaced with <unk>.
        """
        for passage in list_of_word_lists:
            for word_idx in range(len(passage)):
                if passage[word_idx] not in words_seen_during_training:
                    passage[word_idx] = "<unk>"
        return list_of_word_lists

    def compute_perplex(self, list_of_word_lists, probability_dict):
        """
        Computes the perplexity score for each word list in the list_of_word_lists. 

        Arguments:
            list_of_word_lists: a list of word lists
            probability_dict: a unigram or bigram probability dictionary, mapping unigrams
            or bigrams to respective probabilities.
        
        Returns: 
            A list containing the perplexity score for each word list in the list_of_word_lists.
        """
        num_tokens = OpinionSpamModel.count_num_of_tokens(list_of_word_lists)
        perplex_scores = []
        if self.model_type == "unigram":
            for word_list in list_of_word_lists:
                sum_probs = 0
                for word in word_list:
                    prob = probability_dict.get(word)
                    if prob is None:
                        prob = probability_dict.get("<unk>")
                    sum_probs -= math.log(prob)
                result_for_line = math.exp(sum_probs / num_tokens)
                perplex_scores.append(result_for_line)
        elif self.model_type == "bigram":
            for line in list_of_word_lists:
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
        """
        Computes the perplexity score for each word list in the list_of_word_lists. 

        Arguments:
            list_of_word_lists: a list of word lists
            probability_dict: a unigram or bigram probability dictionary, mapping unigrams
            or bigrams to respective probabilities.
        
        Returns: 
            A list containing the perplexity score for each word list in the list_of_word_lists.
        """
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
        Builds a list of predictions, where 0 denotes truthful & 1 denotes deceptive. Iterates
        through both lists and adds the class with the higher perplexity score to the list.

        Arguments:
            complexity_deceptive_list: a list of perplexity scores computed with the deceptive
            probability dictionary.
            complexity_truthful_list: a list of perplexity scores computed with the truthful
            probability dictionary. The indices in this list must align with complexity_deceptive_list,
            such that the socre at each index was calculated upon the same review.
        
        Returns:
            A list of predictions, where 0 denotes truthful & 1 denotes deceptive.
        """
        preds = []
        for i in range(len(complexity_deceptive_list)):
            if complexity_deceptive_list[i] < complexity_truthful_list[i]:
                preds.append(1)
            else:
                preds.append(0)
        return preds
    
    @staticmethod
    def calculate_average_accuracy(truthful_accuracy, num_truthful_predictions, deceptive_accuracy, num_deceptive_predictions):
        """
        Calculates the overall average accuracy, given: the truthful accuracy & number of truthful
        predictions and the deceptive accuracy & number of deceptive predictions.

        Arguments:
            truthful_accuracy: a proportion (0 to 1) representing the proportion of correct 
            truthful predictions
            num_truthful_predictions: the number of predictions made on truthful reviews
            deceptive_accuracy: a proportion (0 to 1) representing the proportion of correct 
            deceptive predictions
            truthful_predictions: the number of predictions made on deceptive reviews
        
        Returns:
            A float (0 to 1) denoting the overall accuracy
        """
        average_accuracy = (
            truthful_accuracy * num_truthful_predictions + 
            deceptive_accuracy * num_deceptive_predictions
        ) / (num_truthful_predictions + num_deceptive_predictions)
        return average_accuracy
    
    @staticmethod
    def get_features_all_reviews(list_of_word_lists, unigram_prob_dict_tru, unigram_prob_dict_dec):
        """
        Produces a list of features to be used for training an sklearn model.

        Arguments:
            list_of_word_lists: a list of word lists
            unigram_prob_dict_tru: dictionary mapping unigrams found in truthful reviews to their 
            respective probabilities
            unigram_prob_dict_dec: dictionary mapping unigrams found in deceptive reviews to their
            respective probabilities

        Returns:
            A list of features to be used for training an sklearn model
        """
        num_tokens = OpinionSpamModel.count_num_of_tokens(list_of_word_lists)
        all_features_numpy = np.asarray([])
        for rev in tqdm(list_of_word_lists):
            rev = " ".join(rev)
            feat_list = feat_extract.get_feature_vector(
                rev, unigram_prob_dict_tru, unigram_prob_dict_dec, num_tokens
            )
            feat_list_numpy = np.asarray(feat_list)
            if all_features_numpy.size == 0:
                all_features_numpy = feat_list_numpy
            else:
                all_features_numpy = np.vstack([all_features_numpy, feat_list_numpy])
        all_features_numpy = preprocessing.normalize(all_features_numpy, axis=0, norm="max")
        return all_features_numpy

    
    def predict(self, validation_data_path=None):
        """
        Encapsulation of logic for obtaining predictions for three types (unigram, bigram and
        Naive Bayes) of opinion spam models on a list of reviews.

        Attributes:
            validation_data_path: Path to .txt file containing validation reviews

        Returns
            A prediction report containing the model's accuracy on the validation reviews
        """
        if validation_data_path is None:
            validation_data_path = self.validation_data_path
        unigram_prob_dict_truthful = self.unigram_truthful
        unigram_prob_dict_deceptive = self.unigram_deceptive
        words_seen_during_training = set(list(unigram_prob_dict_truthful.keys()) + list(unigram_prob_dict_deceptive.keys()))
        labeled_validation_word_lists = OpinionSpamModel.get_passages_from_labeled_txt(validation_data_path)
        truthful_validation_word_lists = []
        deceptive_validation_word_lists = []
        for word_list, label in labeled_validation_word_lists:
            if label == 0:
                truthful_validation_word_lists.append(word_list)
            else:
                deceptive_validation_word_lists.append(word_list)
        truthful_transformed_validation_word_lists = OpinionSpamModel.insert_unks(truthful_validation_word_lists, unigram_prob_dict_truthful.keys())
        deceptive_transformed_validation_word_lists = OpinionSpamModel.insert_unks(deceptive_validation_word_lists, unigram_prob_dict_deceptive.keys())
        prediction_report = dict(keys={})
        if self.model_type in ["unigram", "bigram"]:
            perplex_truthful_tprobs = self.compute_perplex(
                truthful_transformed_validation_word_lists, self.trained_model[self.model_type + "_truthful"]
            )
            perplex_truthful_dprobs = self.compute_perplex(
                truthful_transformed_validation_word_lists, self.trained_model[self.model_type + "_deceptive"]
            )
            perplex_deceptive_tprobs = self.compute_perplex(
                deceptive_transformed_validation_word_lists, self.trained_model[self.model_type + "_truthful"]
            )
            perplex_deceptive_dprobs = self.compute_perplex(
                deceptive_transformed_validation_word_lists, self.trained_model[self.model_type + "_deceptive"]
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
                len(predictions_truthful),
                accuracy_deceptive,
                len(predictions_deceptive),
            )
            prediction_report = {"average_accuracy": "{0:.2%}".format(average_accuracy), "accuracy_truthful": "{0:.2%}".format(accuracy_truthful), "accuracy_deceptive": "{0:.2%}".format(accuracy_deceptive)}
        elif self.model_type in ["NB", "Naive Bayes"]:
            valid_feats_truthful = OpinionSpamModel.get_features_all_reviews(
                truthful_validation_word_lists, unigram_prob_dict_truthful, unigram_prob_dict_deceptive
            )
            valid_feats_deceptive = OpinionSpamModel.get_features_all_reviews(
                deceptive_validation_word_lists, unigram_prob_dict_truthful, unigram_prob_dict_deceptive
            )
            NB_predictions_truthful = self.trained_model.predict(valid_feats_truthful)
            NB_predictions_deceptive = self.trained_model.predict(valid_feats_deceptive)
            NB_accuracy_truthful = OpinionSpamModel.eval_preditions(NB_predictions_truthful, "truthful")
            NB_accuracy_deceptive = OpinionSpamModel.eval_preditions(NB_predictions_deceptive, "deceptive")
            NB_accuracy = OpinionSpamModel.calculate_average_accuracy(
                NB_accuracy_truthful,
                len(NB_predictions_truthful),
                NB_accuracy_deceptive,
                len(NB_predictions_deceptive),
            )
            prediction_report = {"average_accuracy": "{0:.2%}".format(NB_accuracy), "accuracy_truthful": "{0:.2%}".format(NB_accuracy_truthful), "accuracy_deceptive": "{0:.2%}".format(NB_accuracy_deceptive)}
        return prediction_report

if __name__ == "__main__":
    with open(r'config.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    for model in config.keys():
        OpSpamModel = OpinionSpamModel(config[model])
        OpSpamModel.train()
        prediction_report = OpSpamModel.predict()
        print(f"{model}'s accuracy report: {prediction_report}")
        pickle.dump(OpSpamModel, open( config[model]["model_output_path"], "wb" ) )


