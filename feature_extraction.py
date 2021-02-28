import json
import textstat
from sklearn.preprocessing import normalize
import string
import nltk
from collections import Counter
import re
import numpy as np
from empath import Empath
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import math
import numpy as np
from nltk.corpus import stopwords

analyser = SentimentIntensityAnalyzer()
lexicon = Empath()


def remove_stopwords(text):
    """
    Removes stopwords identified by nltk.corpus from the input text.

    Args:
        text: string containing text to remove stopwords from

    Returns:
        The input text with stopwords removed
    """
    word_list = text.split(" ")
    filtered_words = [
        word for word in word_list if word not in stopwords.words("english")
    ]
    new_rev = " ".join(filtered_words)
    return new_rev


def get_sentiment_neg(text):
    """
    Obtains the negative sentiment polarity score from the input text, using
        vaderSentiment analyzer

    Args:
        text: string containing text to calculate the negative sentiment of

    Returns:
        The negative polarity score calculated from the input text
    """
    score = analyser.polarity_scores(text)
    return score["neg"]


def get_sentiment_pos(text):
    """
    Obtains the positive sentiment polarity score from the input text, using
        vaderSentiment analyzer

    Args:
        text: string containing text to calculate the positive sentiment of

    Returns:
        The positive polarity score calculated from the input text
    """
    score = analyser.polarity_scores(text)
    return score["pos"]


def get_readabilty(text):
    """
    Obtains the readability from the input text, using textstat's flesch_reading_ease

    Args:
        text: string containing text to calculate the positive sentiment of

    Returns:
        The positive polarity score calculated from the input text
    """
    readbilty = textstat.flesch_reading_ease(text)
    return readbilty


def get_parts_of_speech(review):
    """
    Obtains the readability from the input text, using textstat's flesch_reading_ease

    Args:
        text: string containing text to calculate the positive sentiment of

    Returns:
        The positive polarity score calculated from the input text
    """
    tokens = nltk.word_tokenize(review.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text, tagset="universal")
    counts = Counter(t for word, t in tagged)

    punc = float((counts["."]))
    noun = float((counts["NOUN"]))
    verb = float((counts["VERB"]))
    pron = float((counts["PRON"]))
    adj = float((counts["ADJ"]))
    return [punc, noun, verb, pron, adj]


def get_uppercase_char_count(text):
    """
    Obtains the number of uppercase characters in the input text

    Args:
        text: string containing text to count the uppercase characters in.

    Returns:
        The number of uppercase characters in the input text
    """
    num_upper = len(re.findall(r"[A-Z]", text))
    return num_upper


def get_exclamation_count(text):
    """
    Obtains the number of exclamation points in the input text

    Args:
        text: string containing text to count the exclamation points in.

    Returns:
        The number of exclamation points in the input text
    """
    c = text.count("!")
    return c


def get_len(text):
    """
    Obtains the length of the input text, in number of characters

    Args:
        text: string containing text to find the length of

    Returns:
        The length of the input text represented as an int
    """
    text_formatted = text.split(" ")
    length_of_text = len(text_formatted)
    return length_of_text


def get_empath_scores(text):
    """
    Obtains empath analysis on the text. Takes the dictionary mapping categories to
    scores, which is produced by passing the text to empath, and returns the scores.

    Args:
        text: string containing text to perform empath analysis on

    Returns:
        A list of empath scores, such that there is a score in the list for each
        of empath's pre-built categories
    """
    empath_dict = lexicon.analyze(text, normalize=True)
    empath_scores = list(empath_dict.values())
    return empath_scores


def get_word_count_lst(text, unigram_prob_dict_truthful, unigram_prob_dict_deceptive):
    """
    Produces a list, where each unique word in the unigram dictionaries is assigned to an
    index, and the value at the index is the number of times that unigram appears in the text.
    Words in the text that do not appear in the unigram dictionaries are ignored.

    Args:
        text: string containing text to find the perplexity of
        unigram_prob_dict_truthful: a dictionary mapping unigrams found in truthful reviews
        to their respective probabilities
        unigram_prob_dict_deceptive: a dictionary mapping unigrams found in deceptive reviews
        to their respective probabilities

    Returns:
        A list representing the occurences of the words in the text (only words that appear
        in the unigram dictionaries are captured)
    """
    training_keys_trutuful = list(unigram_prob_dict_truthful.keys())
    training_keys_deceptive = list(unigram_prob_dict_deceptive.keys())
    all_training_keys = sorted(
        training_keys_trutuful
        + list(set(training_keys_deceptive) - set(training_keys_trutuful))
    )

    word_count_lst = [0] * len(all_training_keys)
    text_lst = text.split(" ")
    for word in text_lst:
        try:
            idx = all_training_keys.index(word)  # unseen word will not be found
            word_count_lst[idx] = word_count_lst[idx] + 1
        except:
            pass

    return word_count_lst


def get_feature_vector(text, unigram_prob_dict_truthful, unigram_prob_dict_deceptive):
    """
    Generates a feature vector based off characteristics of text. Orchestrates calls
    to helper functions that each return a subset of the features to be used.

    Args:
        text: string containing text to generate a feature vector for
        unigram_prob_dict_truthful: a dictionary mapping unigrams in truthful reviews to their
        respecitve probabilities
        unigram_prob_dict_deceptive: a dictionary mapping unigrams in truthful reviews to their
        respecitve probabilities

    Returns:
        A list of features generated from the text
    """
    text_filtered = remove_stopwords(text)
    features = []
    features.append(get_sentiment_pos(text_filtered))
    features.extend(get_empath_scores(text_filtered))
    features.append(get_sentiment_neg(text_filtered))
    features.append(get_readabilty(text))
    features.extend(get_parts_of_speech(text_filtered))
    features.append(get_uppercase_char_count(text))
    features.append(get_exclamation_count(text))
    features.append(get_len(text_filtered))
    features.extend(
        get_word_count_lst(
            text_filtered, unigram_prob_dict_truthful, unigram_prob_dict_deceptive
        )
    )
    floored_features = [0 if feature < 0 else feature for feature in features]
    return floored_features
