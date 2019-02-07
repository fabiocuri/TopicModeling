#!/usr/bin/env python
# coding: utf-8

import os
from spacy.lang.en import English
from hlda.sampler import HierarchicalLDA
import sys

sys.setrecursionlimit(10000)

parser = English()
cwd = os.getcwd()

# Import files

swiss, spanish, both = [], [], []

with open('preprocessed_swiss_fines_only.txt') as f:
    swiss = f.readlines()

swiss = [x.rstrip("\n") for x in swiss]

with open('preprocessed_spanish_fines_only.txt') as f:
    spanish = f.readlines()

spanish = [x.rstrip("\n") for x in spanish]

with open('preprocessed_both_fines_only.txt') as f:
    both = f.readlines()

both = [x.rstrip("\n") for x in both]

swiss = filter_vocabulary(swiss, 0.05)
spanish = filter_vocabulary(spanish, 0.05)
both = filter_vocabulary(both, 0.05)

merged_swiss = ' '.join([' '.join(x) for x in swiss]).split(' ')
merged_spanish = ' '.join([' '.join(x) for x in spanish]).split(' ')
merged_both = ' '.join([' '.join(x) for x in both]).split(' ')

vocab_swiss = list(set(merged_swiss))
vocab_spanish = list(set(merged_spanish))
vocab_both = list(set(merged_both))

vocab_index_swiss = {}
for i, w in enumerate(vocab_swiss):
    vocab_index_swiss[w] = i

vocab_index_spanish = {}
for i, w in enumerate(vocab_spanish):
    vocab_index_spanish[w] = i

vocab_index_both = {}
for i, w in enumerate(vocab_both):
    vocab_index_both[w] = i

new_corpus_swiss = []

for doc in swiss:
    new_doc_swiss = []
    for word in doc:
        word_idx = vocab_index_swiss[word]
        new_doc_swiss.append(word_idx)
    new_corpus_swiss.append(new_doc_swiss)

new_corpus_spanish = []

for doc in spanish:
    new_doc_spanish = []
    for word in doc:
        word_idx = vocab_index_spanish[word]
        new_doc_spanish.append(word_idx)
    new_corpus_spanish.append(new_doc_spanish)

new_corpus_both = []

for doc in both:
    new_doc_both = []
    for word in doc:
        word_idx = vocab_index_both[word]
        new_doc_both.append(word_idx)
    new_corpus_both.append(new_doc_both)

n_samples = 100  # no of iterations for the sampler
alpha = 10.0  # smoothing over level distributions
gamma = 1.0  # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1  # smoothing over topic-word distributions
num_levels = 3  # the number of levels in the tree
display_topics = 100  # the number of iterations between printing a brief summary of the topics so far
n_words = 10  # the number of most probable words to print for each topic after model estimation
with_weights = False  # whether to print the words with the weights

hlda_spanish_1 = HierarchicalLDA(new_corpus_spanish, vocab_spanish, alpha=alpha, gamma=gamma, eta=eta,
                                 num_levels=num_levels)
hlda_spanish_1.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
