#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import string
import nltk

nltk.download('words')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet

sb = SnowballStemmer('english')
words = set(nltk.corpus.words.words())
import os
from spacy.lang.en import English
import sys

sys.setrecursionlimit(10000)

parser = English()
cwd = os.getcwd()

fines_only = True

try:
    swiss_places = pd.read_excel(cwd + '/swiss_places.xlsx', encoding='utf-8', header=None)
    swiss_list_places = list(swiss_places[0])

    swiss_list_places = [x.split() for x in swiss_list_places]

    swiss_list_places_clean = []
    for idx, el in enumerate(swiss_list_places):
        if el[0] == u'-':
            swiss_list_places_clean.append(el[1:])
        elif el[1] == u'Bezirk' or el[1] == u'Verwaltungskreis' or el[1] == u'Wahlkreis':
            swiss_list_places_clean.append(el[2:])
        elif el[1] == u'Arrondissement':
            swiss_list_places_clean.append(el[3:])
        else:
            swiss_list_places_clean.append(el[1:])

    names = []
    for el in swiss_list_places_clean:
        for el1 in el:
            names.append(el1.lower())

    names = list(set(names))

except Exception:

    names = ['']


delete = ['switzerland', 'swiss', 'canton', 'spain', 'catalan', 'spanish']
numbers = list(range(1, 200))
cardinal = list([ordinal(n) for n in numbers])

en_stop = list(set(nltk.corpus.stopwords.words('english'))) + list(
    string.punctuation) + delete + cardinal + numbers + names


dataset = pd.read_excel(cwd + '/Foundations/FoundationsData/Data/SwissData/swiss_foundations.xlsx', encoding='utf-8')
dataset = dataset.fillna('')

corpus = []
for grams in dataset['Mission']:
    corpus.append(grams)

swiss_corpus = corpus

dataset = pd.read_excel(cwd + '/Foundations/FoundationsData/Data/dataframe_translated_into_english.xlsx',
                        encoding='utf-8')
dataset.columns = [u'Actividades', u'Areas_actividad', u'Clasificacion', u'Clasificacion_especifica',
                   u'Codig_actividad', u'Fines', u'Historia',
                   u'Sectores atendidos: Culturales, investigación, medio ambiente o cooperación',
                   u'Sectores atendidos: servicios sociales y salud', u'Tipo_actividades']

dataset = dataset.fillna('')

dataset['Combined'] = list(zip(dataset.Fines, dataset.Actividades, dataset.Codig_actividad, dataset.Historia))

corpus = []
for grams in dataset['Combined']:
    corpus.append((' '.join([w + ' ' for w in grams])).strip())

spanish_corpus = corpus

if fines_only:
    spanish_corpus_fines_only = list(dataset['Fines'])

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    return sb.stem(word)


tags_to_keep = ['JJ', 'NN', 'NNP', 'NNPS', 'NNS']


def prepare_text_for_ML(text):
    tokens = tokenize(text)
    tagged = [token for token in tokens if nltk.pos_tag([token])[0][1] in tags_to_keep]  # Keep certain POS tags.
    tokens = [token for token in tokens if wordnet.synsets(token)]  # Return only English text.
    tokens = [token for token in tokens if token not in en_stop]  # Remove stop-words.
    tokens = [x for x in tokens if not isinstance(x, int)]  # Remove integers.
    tokens = [x for x in tokens if len(x) > 3]  # Remove words with less than 3 letters.
    tokens = [get_lemma(token) for token in tokens]  # Lemmatize words.

    return tokens


preprocessed_swiss, preprocessed_spanish = [], []

for swiss_line in swiss_corpus:
    preprocessed_swiss.append(prepare_text_for_ML(swiss_line))

for spanish_line in spanish_corpus:
    preprocessed_spanish.append(prepare_text_for_ML(spanish_line))

preprocessed_both = preprocessed_swiss + preprocessed_spanish


if fines_only:

    preprocessed_spanish_fines_only = []

    for spanish_line in spanish_corpus_fines_only:
        preprocessed_spanish_fines_only.append(prepare_text_for_ML(spanish_line))

    preprocessed_both_fines_only = preprocessed_swiss + preprocessed_spanish_fines_only

    joined_spanish_data_fines_only, joined_swiss_data_fines_only, joined_both_data_fines_only = [], [], []

    for el in preprocessed_spanish_fines_only:
        joined_spanish_data_fines_only.append(' '.join(el))

    for el in preprocessed_both_fines_only:
        joined_both_data_fines_only.append(' '.join(el))

    for el in preprocessed_swiss:
        joined_swiss_data_fines_only.append(' '.join(el))

if fines_only:
    with open('preprocessed_swiss_fines_only.txt', 'w') as f:
        for item in joined_swiss_data_fines_only:
            f.write("%s\n" % item)

    with open('preprocessed_spanish_fines_only.txt', 'w') as f:
        for item in joined_spanish_data_fines_only:
            f.write("%s\n" % item)

    with open('preprocessed_both_fines_only.txt', 'w') as f:
        for item in joined_both_data_fines_only:
            f.write("%s\n" % item)

