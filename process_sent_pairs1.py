import copy
import csv
import os
import re
import string
import time

import numpy as np
from nltk import pos_tag, SnowballStemmer
from ruwordnet import RuWordNet


preds = []
morph = pymorphy2.MorphAnalyzer()
personal_pronouns = {'он', 'она', 'оно', 'они', 'его', 'ему', 'им', 'нём', 'нем', 'него', 'ее', 'её', 'ей', 'ею' 'ней',
                     'им',
                     'их', 'ими', 'них', 'собой', 'себе', 'себя'}

reports = []
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
SW = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")
punct = list(string.punctuation)
wn = RuWordNet()


def count_fraction_overlap_nouns(sent_l, sent_r):
    cnt = 0
    sent_l_tokens = [morph.parse(t.lower()) for t in nltk.word_tokenize(sent_l)]
    sent_r_tokens = [morph.parse(t.lower()) for t in nltk.word_tokenize(sent_r)]
    unique_noun_lemmas = set()
    for i in sent_l_tokens:
        unique_noun_lemmas.add(i[0].normal_form)

    for j in sent_r_tokens:
        unique_noun_lemmas.add(j[0].normal_form)

    for i in sent_l_tokens:
        for j in sent_r_tokens:
            if i[0].normal_form == j[0].normal_form and i[0].tag.POS == 'NOUN' and j[0].tag.POS == 'NOUN':
                cnt += 1
    return cnt / len(unique_noun_lemmas)


def is_wordform_rep_in_pair(sent_l, sent_r):
    l_tokenized = nltk.word_tokenize(sent_l)
    r_tokenized = nltk.word_tokenize(sent_r)

    l_nouns_stems = [stemmer.stem(w) for w in l_tokenized if morph.tag(w)[0].POS == 'NOUN']
    r_nouns_stems = [stemmer.stem(w) for w in r_tokenized if morph.tag(w)[0].POS == 'NOUN']

    if len(set(l_nouns_stems) & set(r_nouns_stems)) > 0:
        return 1
    else:
        return 0


def is_deriv_in_sent_pair(sent_l, sent_r):
    l_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in nltk.word_tokenize(sent_l) if
                      w not in SW and w not in punct]
    r_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in nltk.word_tokenize(sent_r) if
                      w not in SW and w not in punct]

    l_derivs = set()
    r_derivs = set()
    for w in l_lemmas_WO_SW:
        try:
            for j in wn.get_senses(w)[0].derivations:
                if w != j.name.lower():
                    l_derivs.add(j.name.lower())
        except:
            pass
    for w in r_lemmas_WO_SW:
        try:
            for j in wn.get_senses(w)[0].derivations:
                if w != j.name.lower():
                    r_derivs.add(j.name.lower())
        except:
            pass

    l_derivs_extended = set()
    r_derivs_extended = set()

    for d in l_derivs:
        for sense in wn.get_senses(d):
            for s in sense.derivations:
                if s.name.lower() not in l_lemmas_WO_SW:
                    l_derivs_extended.add(s.name.lower())

    for d in r_derivs:
        for sense in wn.get_senses(d):
            for s in sense.derivations:
                if s.name.lower() not in r_lemmas_WO_SW:
                    r_derivs_extended.add(s.name.lower())

    deriv_in_r = 0
    deriv_in_l = 0
    for i in list(l_derivs_extended):
        if i in r_lemmas_WO_SW:
            deriv_in_r = 1
    for i in list(r_derivs_extended):
        if i in l_lemmas_WO_SW:
            deriv_in_l = 1

    if deriv_in_r == 1 or deriv_in_l == 1:
        return 1
    else:
        return 0


def is_hyponym_in_sent_pair(sent_l, sent_r):
    l_lemmas_WO_SW = []
    l_lemmatized_str = []
    for w in nltk.word_tokenize(sent_l):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            l_lemmatized_str.append(lemma)
            if w not in SW:
                l_lemmas_WO_SW.append(lemma)

    l_lemmatized_str = ' '.join(l_lemmatized_str)

    r_lemmas_WO_SW = []
    r_lemmatized_str = []
    for w in nltk.word_tokenize(sent_r):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            r_lemmatized_str.append(lemma)
            if w not in SW:
                r_lemmas_WO_SW.append(lemma)

    r_lemmatized_str = ' '.join(r_lemmatized_str)

    is_hyponym_in_r = 0
    is_hyponym_in_l = 0

    for w in l_lemmas_WO_SW:
        if is_hyponym_in_r == 1:
            break
        lemmatized_hyponyms = [get_lemmatized_word_phrase(j) for i in get_all_hyponyms(w) for j in i.split(',')]
        for h in lemmatized_hyponyms:
            if h in r_lemmatized_str:
                print(f"Гипоним: {h} от слова: {w}")
                is_hyponym_in_r = 1

    if is_hyponym_in_r == 0:
        for w in r_lemmas_WO_SW:
            if is_hyponym_in_l == 1:
                break
            lemmatized_hyponyms = [get_lemmatized_word_phrase(j) for i in get_all_hyponyms(w) for j in
                                   i.split(',')]
            for h in lemmatized_hyponyms:
                if h in l_lemmatized_str:
                    print(f"Гипоним: {h} от слова: {w}")
                    is_hyponym_in_l = 1

    return is_hyponym_in_r | is_hyponym_in_l


# %%
def is_hypernym_in_sent_pair(sent_l, sent_r):
    l_lemmas_WO_SW = []
    l_lemmatized_str = []
    for w in nltk.word_tokenize(sent_l):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            l_lemmatized_str.append(lemma)
            if w not in SW:
                l_lemmas_WO_SW.append(lemma)

    l_lemmatized_str = ' '.join(l_lemmatized_str)

    r_lemmas_WO_SW = []
    r_lemmatized_str = []
    for w in nltk.word_tokenize(sent_r):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            r_lemmatized_str.append(lemma)
            if w not in SW:
                r_lemmas_WO_SW.append(lemma)

    r_lemmatized_str = ' '.join(r_lemmatized_str)

    is_hypernym_in_r = 0
    is_hypernym_in_l = 0

    for w in l_lemmas_WO_SW:
        if is_hypernym_in_r == 1:
            break
        lemmatized_hypernyms = [get_lemmatized_word_phrase(j) for i in get_all_hypernyms(w) for j in i.split(',')]
        for h in lemmatized_hypernyms:
            if h in r_lemmatized_str:
                print(f"Гиперним: {h} от слова: {w}")
                is_hypernym_in_r = 1

    if is_hypernym_in_r == 0:
        for w in r_lemmas_WO_SW:
            if is_hypernym_in_l == 1:
                break
            lemmatized_hypernyms = [get_lemmatized_word_phrase(j) for i in get_all_hypernyms(w) for j in
                                    i.split(',')]
            for h in lemmatized_hypernyms:
                if h in l_lemmatized_str:
                    print(f"Гиперним: {h} от слова: {w}")
                    is_hypernym_in_l = 1

    return is_hypernym_in_r | is_hypernym_in_l


def is_anph_in_pair(sent_l, sent_r):
    l_tokenized = nltk.word_tokenize(sent_l)
    r_tokenized = nltk.word_tokenize(sent_r)

    possible_anaphora = 0
    l_nouns = []
    anph_pronouns = []

    for i in l_tokenized:
        tag = morph.parse(i)[0].tag
        if tag.POS == 'NOUN':
            l_nouns.append([i, tag.number, tag.gender])

    for j in r_tokenized:
        if morph.parse(j)[0].tag.POS == 'NPRO':
            if 'Anph' in morph.parse(j)[0].tag:
                tag = morph.parse(j)[0].tag
                if j.lower() in personal_pronouns:
                    anph_pronouns.append([j, tag.number, tag.gender])

    for i in l_nouns:
        for j in anph_pronouns:
            if i[1] == j[1] and i[2] == j[2]:
                if len(r_tokenized) > 3:
                    if j[0] in r_tokenized[:3]:
                        possible_anaphora = 1
    return possible_anaphora


def get_lemmatized_word_phrase(word_phrase: str):
    return ' '.join([morph.parse(i.lower().strip())[0].normal_form for i in word_phrase.split(' ')])


def get_all_hypernyms(w):
    hypernyms = []
    if len(wn.get_senses(w)) > 0:
        for i in wn.get_senses(w):
            for j in i.synset.hypernyms:
                _ = re.sub(r"\(.*\)", "", j.title.lower()).strip().split(', ')
                for k in _:
                    if w != k:
                        hypernyms.append(k)
    return hypernyms


def get_all_hyponyms(w):
    hyponyms = []
    if len(wn.get_senses(w)) > 0:
        for i in wn.get_senses(w):
            for j in i.synset.hyponyms:
                _ = re.sub(r"\(.*\)", "", j.title.lower()).strip().split(', ')
                for k in _:
                    if w != k:
                        hyponyms.append(k)
    return hyponyms


predictors = [is_wordform_rep_in_pair, is_deriv_in_sent_pair, is_hyponym_in_sent_pair, is_hypernym_in_sent_pair,
              is_anph_in_pair]
