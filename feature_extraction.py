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

	Arguments:
		text: string containing text to remove stopwords from

	Returns:
		The input text with stopwords removed
	"""
	word_list = text.split(' ')
	filtered_words = [word for word in word_list if word not in stopwords.words('english')]
	new_rev = " ".join(filtered_words)
	return new_rev

def get_sentiment_neg(text): 
	"""
	Obtains the negative sentiment polarity score from the input text, using vaderSentiment analyzer

	Arguments:
		text: string containing text to calculate the negative sentiment of

	Returns:
		The negative polarity score calculated from the input text
	"""
	score = analyser.polarity_scores(text)
	return score['neg']

def get_sentiment_pos(text):
	"""
	Obtains the positive sentiment polarity score from the input text, using vaderSentiment analyzer

	Arguments:
		text: string containing text to calculate the positive sentiment of

	Returns:
		The positive polarity score calculated from the input text
	"""
	score = analyser.polarity_scores(text)
	return score['pos']


def get_readabilty(text):
	"""
	Obtains the readability from the input text, using textstat's flesch_reading_ease

	Arguments:
		text: string containing text to calculate the positive sentiment of

	Returns:
		The positive polarity score calculated from the input text
	"""
	readbilty = textstat.flesch_reading_ease(text)
	return readbilty


def get_parts_of_speech(review):
	"""
	Obtains the readability from the input text, using textstat's flesch_reading_ease

	Arguments:
		text: string containing text to calculate the positive sentiment of

	Returns:
		The positive polarity score calculated from the input text
	"""
	tokens = nltk.word_tokenize(review.lower())
	text = nltk.Text(tokens)
	tagged = nltk.pos_tag(text, tagset='universal')
	counts = Counter(t for word,t in tagged)

	punc = float((counts['.']))
	noun = float((counts['NOUN']))
	verb = float((counts['VERB']))
	pron = float((counts['PRON']))
	adj = float((counts['ADJ']))
	return [punc,noun, verb, pron, adj]


def get_uppercase_char_count(text):
	"""
	Obtains the number of uppercase characters in the input text

	Arguments:
		text: string containing text to count the uppercase characters in.

	Returns:
		The number of uppercase characters in the input text
	"""
   	num_upper = len(re.findall(r'[A-Z]', text))
   	return num_upper

def get_exclamation_count(text):
	"""
	Obtains the number of exclamation points in the input text

	Arguments:
		text: string containing text to count the exclamation points in.

	Returns:
		The number of exclamation points in the input text
	"""
	c = text.count("!")
	return c

def get_len(text):
	"""
	Obtains the length of the input text, in number of characters

	Arguments:
		text: string containing text to find the length of

	Returns:
		The length of the input text
	"""
	text_formatted = text.split(" ")
	length_of_text = len(text_formatted)
	return length_of_text


def get_empath_results(review):
	empath_dict = lexicon.analyze(review, normalize=True)
	vals =list(empath_dict.values())
	return vals


def compute_perplex_unigram(review, unigram_prob_dict, n):

    review_lst = review.split(" ")
    sum_probs = 0
    for word in review_lst:
        prob = unigram_prob_dict.get(word)
        if prob == None:
            prob = unigram_prob_dict.get("<unk>")
        sum_probs -= math.log(prob)

    result_for_line = math.exp(sum_probs/n)
    
    return [result_for_line]



def word_count_feats(review, unigram_prob_dict_tru, unigram_prob_dict_dec):

	uni_keys_tru = list(unigram_prob_dict_tru.keys())
	uni_keys_dec = list(unigram_prob_dict_dec.keys())
	num_unks = 0
	training_keys = sorted(uni_keys_tru + list(set(uni_keys_dec) - set(uni_keys_tru)))

	word_count_feats = [0]*len(training_keys)

	review_list = review.split(" ")
	for word in review_list: 
		try: 
			idx = training_keys.index(word)
			word_count_feats[idx] = word_count_feats[idx] + 1
		except: 
			num_unks += 1

	return word_count_feats


def get_feature_vector(review, unigram_prob_dict_tru, unigram_prob_dict_dec, n):

	review_filtered = remove_stopwords(review)

	feats = (get_sentiment_pos(review_filtered))

	feats.extend(get_empath_results(review_filtered))
	
	feats.append(get_sentiment_neg(review_filtered))
	feats.append(get_readabilty(review))
	feats.extend(get_parts_of_speech(review_filtered))
	feats.append(get_uppercase_char_count(review))
	feats.append(get_exclamation_count(review))
	feats.append(get_len(review_filtered))
	
	feats.extend(word_count_feats(review_filtered, unigram_prob_dict_tru, unigram_prob_dict_dec))

	feats.extend(compute_perplex_unigram(review_filtered, unigram_prob_dict_dec, n))
	feats.extend(compute_perplex_unigram(review_filtered, unigram_prob_dict_tru, n))

	feats2 = [0 if i < 0 else i for i in feats]

	return (feats2)




