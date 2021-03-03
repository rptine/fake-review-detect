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


class OpSpamModelBuilder:
    """
    Opinion Spam Model Builder

    Encapsulation of training and evaluating three types (unigram, bigram and Naive Bayes) of
    opinion spam models. Opinion spam models predict whether text reviews are truthful or
    deceptive.

    Attributes:
        model_type: one of unigram, bigram or Naive Bayes
        training_data_path: Path to .txt file containing training data
        validation_data_path: Path to .txt file containing validation data
    """

    def __init__(
        self,
        config=None,
        model_type=None,
        training_data_path=None,
        validation_data_path=None,
    ):
        """
        Initializes OpSpamModelBuilder with a model type, training_data_path and validation_data_path.

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
            (passage.strip().split(" "), label)
            for passage, label in list_of_labeled_passages
        ]  # strip to remove newline chars
        return list_of_labeled_passages

    @staticmethod
    def count_num_of_tokens(list_of_token_lists):
        """
        Finds the total number of tokens contained in a list of strings.

        Args:
            list_of_token_lists: a list of lists of words, where each word is a string

        Returns:
            The total number of tokens in the list of strings, where a token is a
            non-unique word within the string.
        """
        num_tokens = 0
        for line in list_of_token_lists:
            num_tokens += len(line)
        return num_tokens

    @staticmethod
    def build_unigram_count_dict(list_of_token_lists):
        """
        Produces a dictionary of unigram counts, where each unique word in the list of
        word lists is a key, and the key's corresponding value is the number of occurrences
        of the word.

        Args:
            list_of_token_lists: a list of token lists

        Returns:
            A dictionary of unigram counts, mapping unique words in list_of_token_lists
            to the number of occurrences.
        """
        unigram_count_dict = defaultdict(int)
        for line in list_of_token_lists:
            for word in line:
                unigram_count_dict[word] += 1
        unigram_count_dict["<unk>"] = 1
        return unigram_count_dict

    @staticmethod
    def build_bigram_count_dict(list_of_token_lists, unigram_count_dict):
        """
        Produces a dictionary of bigram counts, where each unique pair of successive words in
        list_of_token_lists is a key, and the key's corresponding value is the number of occurrences
        of the pair of words.

        Args:
            list_of_token_lists: a list of token lists

        Returns:
            A dictionary of bigram counts, mapping unique pairs of successive words in
            list_of_token_lists to the number of occurrences.
        """
        bigram_count_dict = defaultdict(int)
        words_seen_once = defaultdict(int)
        for line in list_of_token_lists:
            for i in range(len(line) - 1):
                if (line[i] in words_seen_once.keys()) and (
                    line[i + 1] in words_seen_once.keys()
                ):
                    bigram_count_dict[line[i] + " " + line[i + 1]] += 1
                elif (line[i] in words_seen_once.keys()) and (
                    not line[i + 1] in words_seen_once.keys()
                ):
                    bigram_count_dict[line[i] + " " + "<unk>"] += 1
                    words_seen_once[line[i + 1]] = 0
                elif (not line[i] in words_seen_once.keys()) and (
                    line[i + 1] in words_seen_once.keys()
                ):
                    bigram_count_dict["<unk>" + " " + line[i]] += 1
                    words_seen_once[line[i]] = 0
                else:
                    words_seen_once[line[i]] = 0
                    words_seen_once[line[i + 1]] = 0
        bigram_count_dict["<unk> <unk>"] = 1
        return bigram_count_dict

    @staticmethod
    def get_unigram_prob_dict(unigram_count_dict, list_of_token_lists, num_tokens):
        """
        Produces a dictionary of unigram probabilities for all unigrams in the unigram_count_dict.

        Args:
            unigram_count_dict: a dictionary mapping unigrams (single words) to their respective
            counts
            list_of_token_lists: a list of token lists
            num_tokens: the number of tokens in the list of word lists

        Returns:
            A dictionary mapping unigrams to their respective probabilities for all unigrams in the
            unigram_count_dict. The probability is calculated from the relative the counts of each
            unigram and incorporates plus-one-smoothing.
        """
        # Add the number of unigrams to num_tokens, since 1 is added to the numerator for each unigram
        num_tokens_smoothed = num_tokens + len(unigram_count_dict.keys())
        unigram_prob_dict = {
            k: ((v + 1) / num_tokens_smoothed) for k, v in unigram_count_dict.items()
        }
        return unigram_prob_dict

    @staticmethod
    def get_bigram_prob_dict(bigram_count_dict, unigram_count_dict, unigram_prob_dict):
        """
        Produces a dictionary of bigram probabilities for all bigrams in the bigram_count_dict.

        Args:
            bigram_count_dict: a dictionary mapping bigrams (pairs of successive words) to their
            respective counts
            unigram_count_dict: a dictionary mapping unigrams (single words) to their respective
            counts
            unigram_count_dict: a dictionary mapping unigrams (single words) to their respective
            probabilities.
            All three of the above dictionaries should be derived from the same source text.

        Returns:
            A dictionary mapping bigrams to their respective probabilities, for all bigrams in the
            bigram_count_dict. The probability is calculated from the relative the counts of each
            bigram and incorporates plus-one-smoothing.
        """
        bigram_prob_dict = defaultdict(int)
        for bigram_key in bigram_count_dict.keys():
            bigram_key_count = bigram_count_dict.get(bigram_key)
            first_word = bigram_key.split(" ")[0]
            first_word_count = unigram_count_dict.get(first_word)
            # cap first word count at 10 to avoid assigning bigrams with common first words low probs
            if first_word_count > 10:
                first_word_count = 10
            cond_prob = bigram_key_count / first_word_count
            bigram_prob = cond_prob * unigram_prob_dict.get(first_word)
            bigram_prob_dict[bigram_key] = bigram_prob
        return bigram_prob_dict

    def train(self, training_data_path=None):
        """
        Encapsulation of logic for training three types (unigram, bigram and Naive Bayes) of
        opinion spam models.

        Args:
            training_data_path: Path to .txt file containing training data

        Returns
            OpSpamModelBuilder object with the attribute trained_model containing a reference to
            to a model trained on the data obtained from the training_data_path.
        """
        if training_data_path is None:
            training_data_path = self.training_data_path
        labeled_train_token_lists = OpSpamModelBuilder.get_passages_from_labeled_txt(
            training_data_path
        )
        # split training word lists into truthful and deceptive sets
        truthful_train_token_lists = []
        deceptive_train_token_lists = []
        for token_list, label in labeled_train_token_lists:
            if label == 0:
                truthful_train_token_lists.append(token_list)
            else:
                deceptive_train_token_lists.append(token_list)
        unigram_count_dict_truthful = OpSpamModelBuilder.build_unigram_count_dict(
            truthful_train_token_lists
        )
        unigram_count_dict_deceptive = OpSpamModelBuilder.build_unigram_count_dict(
            deceptive_train_token_lists
        )
        num_tokens_truthful = OpSpamModelBuilder.count_num_of_tokens(
            truthful_train_token_lists
        )
        num_tokens_deceptive = OpSpamModelBuilder.count_num_of_tokens(
            deceptive_train_token_lists
        )
        unigram_prob_dict_truthful = OpSpamModelBuilder.get_unigram_prob_dict(
            unigram_count_dict_truthful, truthful_train_token_lists, num_tokens_truthful
        )
        unigram_prob_dict_deceptive = OpSpamModelBuilder.get_unigram_prob_dict(
            unigram_count_dict_deceptive,
            deceptive_train_token_lists,
            num_tokens_deceptive,
        )
        self.unigram_truthful = unigram_prob_dict_truthful
        self.unigram_deceptive = unigram_prob_dict_deceptive
        model = {
            "unigram_truthful": unigram_prob_dict_truthful,
            "unigram_deceptive": unigram_prob_dict_deceptive,
        }
        if (
            self.model_type == "bigram"
        ):  # add bigram dictionaries to model if model type is bigram
            bigram_count_dict_truthful = OpSpamModelBuilder.build_bigram_count_dict(
                truthful_train_token_lists, unigram_count_dict_truthful
            )
            bigram_count_dict_deceptive = OpSpamModelBuilder.build_bigram_count_dict(
                deceptive_train_token_lists, unigram_count_dict_deceptive
            )
            bigram_prob_dict_truthful = OpSpamModelBuilder.get_bigram_prob_dict(
                bigram_count_dict_truthful,
                unigram_count_dict_truthful,
                unigram_prob_dict_truthful,
            )
            bigram_prob_dict_deceptive = OpSpamModelBuilder.get_bigram_prob_dict(
                bigram_count_dict_deceptive,
                unigram_count_dict_deceptive,
                unigram_prob_dict_deceptive,
            )
            model["bigram_truthful"] = bigram_prob_dict_truthful
            model["bigram_deceptive"] = bigram_prob_dict_deceptive
        elif self.model_type in ["NB", "Naive Bayes"]:
            training_feat_lst_truthful = [
                OpSpamModelBuilder.get_features(
                    truthful_train_token_lists[0],
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
            ]
            for token_list in tqdm(truthful_train_token_lists[1:]):
                training_feat_truthful = OpSpamModelBuilder.get_features(
                    token_list,
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
                training_feat_lst_truthful = np.vstack(
                    [training_feat_lst_truthful, training_feat_truthful]
                )
                training_feat_lst_truthful = preprocessing.normalize(
                    training_feat_lst_truthful, axis=0, norm="max"
                )
            training_feat_lst_deceptive = [
                OpSpamModelBuilder.get_features(
                    deceptive_train_token_lists[0],
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
            ]
            for token_list in tqdm(deceptive_train_token_lists[1:]):
                training_feat_deceptive = OpSpamModelBuilder.get_features(
                    token_list,
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
                training_feat_lst_deceptive = np.vstack(
                    [training_feat_lst_deceptive, training_feat_deceptive]
                )
                training_feat_lst_deceptive = preprocessing.normalize(
                    training_feat_lst_deceptive, axis=0, norm="max"
                )
            labels_truthful = np.zeros(len(truthful_train_token_lists))
            labels_deceptive = np.ones(len(deceptive_train_token_lists))
            all_feats_train = np.vstack(
                (training_feat_lst_deceptive, training_feat_lst_truthful)
            )
            all_labels_train = np.concatenate((labels_deceptive, labels_truthful))
            model = naive_bayes.MultinomialNB()
            model.fit(all_feats_train, all_labels_train)
        self.trained_model = model
        return model

    @staticmethod
    def insert_unks(list_of_token_lists, words_seen_during_training):
        """
        Replaces strings in the list_of_token_lists that do not appear as keys in
        words_seen_during_training with the string <unk>.

        Args:
            list_of_token_lists: a list of token lists
            words_seen_during_training: a list of words seen during training

        Returns:
            A modified version of the input list_of_token_lists, where each word not present in
            words_seen_during_training is replaced with <unk>.
        """
        for passage in list_of_token_lists:
            for word_idx in range(len(passage)):
                if passage[word_idx] not in words_seen_during_training:
                    passage[word_idx] = "<unk>"
        return list_of_token_lists

    @staticmethod
    def compute_perplex(model_type, token_list, probability_dict):
        """
        Computes the perplexity score for the token_list

        Args:
            model_type: a string indiciating the model type, either "unigram" or "bigram"
            token_list: a list of tokens to compute the probability score of
            probability_dict: a unigram or bigram probability dictionary, mapping unigrams
            or bigrams to respective probabilities.

        Returns:
            The perplexity score for the token_list.
        """
        sum_probs = 0
        if model_type == "unigram":
            for word in token_list:
                prob = probability_dict.get(word)
                if prob is None:
                    prob = probability_dict.get("<unk>")
                sum_probs -= math.log(prob)
            result_for_token_list = math.exp(sum_probs / len(token_list))
            return result_for_token_list
        elif model_type == "bigram":
            for i in range(len(token_list) - 1):
                prob = 0
                bigram = str(token_list[i] + " " + token_list[i + 1])
                prob = probability_dict.get(bigram)
                if prob is None:
                    prob = probability_dict.get("<unk> <unk>")
                sum_probs -= math.log(prob)
            result_for_token_list = math.exp(sum_probs / len(token_list))
            return result_for_token_list

    @staticmethod
    def eval_preditions(prediction_list, list_actuals):
        """
        Computes the perplexity score for each word list in the list_of_token_lists.

        Args:
            list_of_token_lists: a list of token lists
            probability_dict: a unigram or bigram probability dictionary, mapping unigrams
            or bigrams to respective probabilities.

        Returns:
            A list containing the perplexity score for each word list in the list_of_token_lists.
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
        through both lists and adds the class with the lower perplexity score to the output list.

        Args:
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
    def calculate_average_accuracy(
        truthful_accuracy,
        num_truthful_predictions,
        deceptive_accuracy,
        num_deceptive_predictions,
    ):
        """
        Calculates the overall average accuracy, given: the truthful accuracy & number of truthful
        predictions and the deceptive accuracy & number of deceptive predictions.

        Args:
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
            truthful_accuracy * num_truthful_predictions
            + deceptive_accuracy * num_deceptive_predictions
        ) / (num_truthful_predictions + num_deceptive_predictions)
        return average_accuracy

    @staticmethod
    def get_features(token_list, unigram_prob_dict_tru, unigram_prob_dict_dec):
        """
        Produces a list of features to be used for training an sklearn model.

        Args:
            token_list: a list of tokens to produce features for
            unigram_prob_dict_tru: dictionary mapping unigrams found in truthful reviews to their
            respective probabilities
            unigram_prob_dict_dec: dictionary mapping unigrams found in deceptive reviews to their
            respective probabilities

        Returns:
            A list of features sourced from the token_list, to be used for training an sklearn model
        """
        all_features_numpy = np.asarray([])
        token_list = " ".join(token_list)
        feat_list = feat_extract.get_feature_vector(
            token_list, unigram_prob_dict_tru, unigram_prob_dict_dec
        )
        feat_list_numpy = np.asarray(feat_list)
        return feat_list_numpy

    def predict(self, text):
        """
        Encapsulation of logic for obtaining a single prediction for three types (unigram, bigram and
        Naive Bayes) of opinion spam models on a single prediction.

        Args:
            validation_data_path: Path to .txt file containing validation reviews

        Returns
            A prediction report containing the model's accuracy on the validation reviews
        """
        if self.model_type in ["unigram", "bigram"]:
            perplex_truthful_tprobs = self.compute_perplex(
                text,
                self.trained_model[self.model_type + "_truthful"],
            )
            perplex_truthful_dprobs = self.compute_perplex(
                text,
                self.trained_model[self.model_type + "_deceptive"],
            )
            perplex_deceptive_tprobs = self.compute_perplex(
                text,
                self.trained_model[self.model_type + "_truthful"],
            )
            perplex_deceptive_dprobs = self.compute_perplex(
                text,
                self.trained_model[self.model_type + "_deceptive"],
            )
            prediction = 0
        elif self.model_type in ["NB", "Naive Bayes"]:
            valid_feats_truthful = OpSpamModelBuilder.get_features(
                text,
                unigram_prob_dict_truthful,
                unigram_prob_dict_deceptive,
            )
            valid_feats_deceptive = OpSpamModelBuilder.get_features(
                text,
                unigram_prob_dict_truthful,
                unigram_prob_dict_deceptive,
            )
            prediction = self.trained_model.predict(text)
        return prediction

    def predict_on_list(self, validation_data_path=None):
        """
        Encapsulation of logic for obtaining accuracy values for three types (unigram, bigram and
        Naive Bayes) of opinion spam models on predicting on a list of reviews.

        Args:
            validation_data_path: Path to .txt file containing validation reviews

        Returns
            A prediction report containing the model's accuracy on the validation reviews
        """
        if validation_data_path is None:
            validation_data_path = self.validation_data_path
        unigram_prob_dict_truthful = self.unigram_truthful
        unigram_prob_dict_deceptive = self.unigram_deceptive
        words_seen_during_training = set(
            list(unigram_prob_dict_truthful.keys())
            + list(unigram_prob_dict_deceptive.keys())
        )
        labeled_validation_token_lists = (
            OpSpamModelBuilder.get_passages_from_labeled_txt(validation_data_path)
        )
        truthful_validation_token_lists = []
        deceptive_validation_token_lists = []
        for token_list, label in labeled_validation_token_lists:
            if label == 0:
                truthful_validation_token_lists.append(token_list)
            else:
                deceptive_validation_token_lists.append(token_list)
        truthful_transformed_validation_token_lists = OpSpamModelBuilder.insert_unks(
            truthful_validation_token_lists, unigram_prob_dict_truthful.keys()
        )
        deceptive_transformed_validation_token_lists = OpSpamModelBuilder.insert_unks(
            deceptive_validation_token_lists, unigram_prob_dict_deceptive.keys()
        )
        prediction_report = dict(keys={})
        if self.model_type in ["unigram", "bigram"]:
            perplex_truthful_tprobs = [
                OpSpamModelBuilder.compute_perplex(
                    self.model_type,
                    token_list,
                    self.trained_model[self.model_type + "_truthful"],
                )
                for token_list in truthful_transformed_validation_token_lists
            ]
            perplex_truthful_dprobs = [
                OpSpamModelBuilder.compute_perplex(
                    self.model_type,
                    token_list,
                    self.trained_model[self.model_type + "_deceptive"],
                )
                for token_list in truthful_transformed_validation_token_lists
            ]
            perplex_deceptive_tprobs = [
                OpSpamModelBuilder.compute_perplex(
                    self.model_type,
                    token_list,
                    self.trained_model[self.model_type + "_truthful"],
                )
                for token_list in deceptive_transformed_validation_token_lists
            ]
            perplex_deceptive_dprobs = [
                OpSpamModelBuilder.compute_perplex(
                    self.model_type,
                    token_list,
                    self.trained_model[self.model_type + "_deceptive"],
                )
                for token_list in deceptive_transformed_validation_token_lists
            ]
            predictions_truthful = OpSpamModelBuilder.make_predictions(
                perplex_truthful_dprobs, perplex_truthful_tprobs
            )
            predictions_deceptive = OpSpamModelBuilder.make_predictions(
                perplex_deceptive_dprobs, perplex_deceptive_tprobs
            )
            accuracy_truthful = OpSpamModelBuilder.eval_preditions(
                predictions_truthful, "truthful"
            )
            accuracy_deceptive = OpSpamModelBuilder.eval_preditions(
                predictions_deceptive, "deceptive"
            )
            average_accuracy = OpSpamModelBuilder.calculate_average_accuracy(
                accuracy_truthful,
                len(predictions_truthful),
                accuracy_deceptive,
                len(predictions_deceptive),
            )
            prediction_report = {
                "average_accuracy": "{0:.2%}".format(average_accuracy),
                "accuracy_truthful": "{0:.2%}".format(accuracy_truthful),
                "accuracy_deceptive": "{0:.2%}".format(accuracy_deceptive),
            }
        elif self.model_type in ["NB", "Naive Bayes"]:
            # Features from truthful texts
            valid_feats_lst_truthful = [
                OpSpamModelBuilder.get_features(
                    truthful_validation_token_lists[0],
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
            ]
            for token_list in tqdm(truthful_validation_token_lists[1:]):
                valid_feat_truthful = OpSpamModelBuilder.get_features(
                    token_list,
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
                valid_feats_lst_truthful = np.vstack(
                    [valid_feats_lst_truthful, valid_feat_truthful]
                )
            valid_feats_lst_truthful = preprocessing.normalize(
                valid_feats_lst_truthful, axis=0, norm="max"
            )
            # Features from deceptive texts
            valid_feats_lst_deceptive = [
                OpSpamModelBuilder.get_features(
                    deceptive_validation_token_lists[0],
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
            ]
            for token_list in tqdm(deceptive_validation_token_lists[1:]):
                valid_feat_deceptive = OpSpamModelBuilder.get_features(
                    token_list,
                    unigram_prob_dict_truthful,
                    unigram_prob_dict_deceptive,
                )
                valid_feats_lst_deceptive = np.vstack(
                    [valid_feats_lst_deceptive, valid_feat_deceptive]
                )
                valid_feats_lst_deceptive = preprocessing.normalize(
                    valid_feats_lst_deceptive, axis=0, norm="max"
                )
            NB_predictions_truthful = self.trained_model.predict(
                valid_feats_lst_truthful
            )
            NB_predictions_deceptive = self.trained_model.predict(
                valid_feats_lst_deceptive
            )
            NB_accuracy_truthful = OpSpamModelBuilder.eval_preditions(
                NB_predictions_truthful, "truthful"
            )
            NB_accuracy_deceptive = OpSpamModelBuilder.eval_preditions(
                NB_predictions_deceptive, "deceptive"
            )
            NB_accuracy = OpSpamModelBuilder.calculate_average_accuracy(
                NB_accuracy_truthful,
                len(NB_predictions_truthful),
                NB_accuracy_deceptive,
                len(NB_predictions_deceptive),
            )
            prediction_report = {
                "average_accuracy": "{0:.2%}".format(NB_accuracy),
                "accuracy_truthful": "{0:.2%}".format(NB_accuracy_truthful),
                "accuracy_deceptive": "{0:.2%}".format(NB_accuracy_deceptive),
            }
        return prediction_report


if __name__ == "__main__":
    with open(r"config.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    for model in config.keys():
        OpSpamModel = OpSpamModelBuilder(config[model])
        OpSpamModel.train()
        prediction_report = OpSpamModel.predict_on_list()
        print(f"{model}'s accuracy report: {prediction_report}")
        pickle.dump(OpSpamModel, open(config[model]["model_output_path"], "wb"))
