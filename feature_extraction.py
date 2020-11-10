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


def remove_stopwords(review):
	word_list = review.split(' ')
	filtered_words = [word for word in word_list if word not in stopwords.words('english')]
	new_rev = " ".join(filtered_words)
	return new_rev

def get_sentiment_neg(review): 
	score = analyser.polarity_scores(review)
	return [score['neg']] 

def get_sentiment_pos(review):
	score = analyser.polarity_scores(review)
	return [score['pos']]


def get_readabilty(review): 
	readbilty = textstat.flesch_reading_ease(review)
	return [readbilty]


def get_parts_of_speech(review):
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


def uppercase_char_count(review):
   	y = len(re.findall(r'[A-Z]',review))
   	return [y]

def exclamation_count(review):
	c = review.count("!")
	return [c]

def get_len(review):
	rev_list = review.split(" ")
	return [len(rev_list)]


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
	
	feats.extend(get_sentiment_neg(review_filtered))
	feats.extend(get_readabilty(review))
	feats.extend(get_parts_of_speech(review_filtered))
	feats.extend(uppercase_char_count(review))
	feats.extend(exclamation_count(review))
	feats.extend(get_len(review_filtered))
	
	feats.extend(word_count_feats(review_filtered, unigram_prob_dict_tru, unigram_prob_dict_dec))

	feats.extend(compute_perplex_unigram(review_filtered, unigram_prob_dict_dec, n))
	feats.extend(compute_perplex_unigram(review_filtered, unigram_prob_dict_tru, n))

	feats2 = [0 if i < 0 else i for i in feats]

	return (feats2)




