# fake-review-detect
Three models, unigram, bigram, and Naive Bayes, for detecting reviews that contain opinion spam.

## Effectiveness
### Unigram Model
Average Overall Accuracy: 88.24% <br>
Accuracy On Truthful: 77.95% <br>
Acuracy On Deceptive: 98.44% <br>
### Bigram Model
Average Overall Accuracy: 89.02% <br>
Accuracy On Truthful: 88.19% <br>
Acuracy On Deceptive: 89.84% <br>
## Naive Bayes
Average Overall Accuracy: 90.59% <br>
Accuracy On Truthful: 84.25% <br>
Acuracy On Deceptive: 96.88% <br>

## Implementation Details
### Probability Calculation + Smoothing
Unigram Model:
To calculate the probability of each unigram, without smoothing, we divide the number of occurences of each unigram by the total number of unique tokens in the training corpus: <br>
```unigram_prob = Count(Unigram) / Count(Total Unique Tokens)```<br>
This introduces sparsity issues since many unigrams will have a probability close to 0; this is because the majority of unigrams appear with low frequency. We use plus-one smoothing to assuage this issue. To implement plus-one smoothing, we add one to the count of each unigram in our probability calculation; since we increased the count of each unigram by one, add the cardinality of the unigram dictionary to the denomniator in our calculation:
```unigram_prob_smoothed = Count(Unigram) + 1 / Count(Total Unique Tokens) + Size of Unigram Dictionary```

Bigram Model:
To calculate the probability of each bigram, we first find the conditional probability of the bigram occuring, given the first word occurs:
```cond_prob = Count(Bigram) / Count(First Word)```
We then multiply this conditional probability by the probability of the first word occuring (which is obtained from the unigram probability dictionary):
```bigram_prob = cond_prob * unigram_prob_dict[First Word]```
This incorporates plus-one smoothing because it leverages the unigram probabiltiy dictionary which contains probailities that has already been smoothed.
We also cap the count of the first word at 10 because a small portion of unigrams have large frequencies. This causes a drag on the conditional probability and reduces the probability of bigrams that contain these common unigrams below a reasonable level.


### UNK Handling
Unigram Model:
```("<unk>", 1)``` is added to the unigram count dictionary and the probability of this unigram is calculated with the same methodology used for the other unigrams. During prediction, when an unseen word (a.k.a. a word that is not contained in the trained model's unigram dictionary) is encountered, the unk's probability is used in the place of the unseen word's probability.

Bigram Model:
Words seen only once during training are replaced with ```"<unk>"``` in the bigram count dictionary.  ```("<unk> <unk>", 1)``` is added to the bigram count dictionary and the probability of this bigram is calculated with the same methodology used for the other bigrams.
### Perpexlity
### Classification 

### Naive Bayes
### Features

## Acknowledgments
Negative Deceptive Opinion Spam[https://www.aclweb.org/anthology/N13-1053.pdf]