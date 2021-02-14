# fake-review-detect
We built three models for detecting spam reviews.  Two of the models, unigram and bigram, are language models and the third is a Naive Bayes model.

## Language Model
### Unigram Model
### Bigram Model
### Smoothing
### UNK Handling
Unigram Model:
```("<unk>", 1)``` is added to the unigram count dictionary and the probability of this unigram is calculated with the same methodology used for the other unigrams. During prediction, when an unseen word (a.k.a. a word that is not contained in the trained model's unigram dictionary) is encountered, the unk's probability is used in the place of the unseen word's probability.

Bigram Model:
Words seen only once during training are replaced with ```"<unk>"``` in the bigram count dictionary.  ```("<unk> <unk>", 1)``` is added to the bigram count dictionary and the probability of this bigram is calculated with the same methodology used for the other bigrams.
### Perpexlity
### Classification 

## Naive Bayes
### NB Model
### Features

## Acknowledgments
Negative Deceptive Opinion Spam[https://www.aclweb.org/anthology/N13-1053.pdf]