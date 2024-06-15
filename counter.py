import multiprocessing
import os
import string
from concurrent.futures import ProcessPoolExecutor
import regex as re
import numpy as np
import pymorphy2
import csv
import random
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
from ruwordnet import RuWordNet
from nltk.corpus import stopwords

'''
# Описание переменных 
– re_translation - хранит регулярное выражение для посика заголовков файлов текстов переводов с англ на рус
– files - хранит все заголовки файлов с текстами
– rus_trans_texts - список текстов переводов с англ на рус
– stemmer - хранит объект стеммера, пригодного для русского языка
– morph – хранит объект морфологического анализатора
– wn – хранит объект модели ORM для связи с БД русского WordNet
– SW - множество русских стопслов
– punct - список основных символов пунктуации

# Описание функций
– is_wordform_rep_in_pair - проверяет, есть ли в паре предложений повторы словоформ (солнце - солнца), возвращает 1 или 0
– is_deriv_in_sent_pair - проверяет, есть ли в паре предложений производная лексика (солнце - солнечный), возвращает 1 или 0
– is_hyponym_in_sent_pair - проверяет, есть ли в паре предложений гипонимы (слова спаржа для сочетания травянистое растение), возвращает 1 или 0
– is_hypernym_in_sent_pair - проверяет, есть ли в паре предложений гиперонимы (сочетание травянистое растение для слова спаржа), возвращает 1 или 0
– is_anph_in_pair - проверяет, является ли пара предложений «кандидатом» на наличие анафоры в ней 
(Я пошел в магазин. Он оказался закрыт.), возвращает 1 или 0 
– main – точка входа, считывание файлов, создание датасета с использованием асинхронности 
'''

stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()
wn = RuWordNet()

SW = set(stopwords.words('russian'))
punct = list(string.punctuation)


def is_wordform_rep_in_pair(sent_pair: list[list[str]]):
    l = sent_pair[0]
    r = sent_pair[1]

    l_tokenized = word_tokenize(l)
    r_tokenized = word_tokenize(r)

    l_nouns_stems = [stemmer.stem(w) for w in l_tokenized if morph.tag(w)[0].POS == 'NOUN']
    r_nouns_stems = [stemmer.stem(w) for w in r_tokenized if morph.tag(w)[0].POS == 'NOUN']

    if len(set(l_nouns_stems) & set(r_nouns_stems)) > 0:
        return 1
    else:
        return 0


# %%
def is_deriv_in_sent_pair(sent_pair: list[list[str]]):
    l = sent_pair[0]
    r = sent_pair[1]

    l_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(l) if w not in SW and w not in punct]
    r_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(r) if w not in SW and w not in punct]

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

    # print(l_derivs_extended)

    deriv_in_r = 0
    deriv_in_l = 0
    # print(l_derivs_extended)
    # print(r_derivs_extended)
    for i in list(l_derivs_extended):
        if i in r_lemmas_WO_SW:
            # print(i)
            deriv_in_r = 1
    for i in list(r_derivs_extended):
        if i in l_lemmas_WO_SW:
            # print(i)
            deriv_in_l = 1

    # print(set(l_derivs) & set(r_derivs))
    # if len(set(l_derivs) & set(r_derivs)) > 0:
    if deriv_in_r == 1 or deriv_in_l == 1:
        return 1
    else:
        return 0

    # is_deriv = 0
    # for _ in derivs_sent_1:
    #     if _ in sent2
    #         is_deriv = 1
    #         break
    #
    # return is_deriv


# %%
def is_hyponym_in_sent_pair(sent_pair: list[list[str]]):
    l = sent_pair[0]
    r = sent_pair[1]

    l_lemmas_WO_SW = []
    l_lemmatized_str = []
    for w in word_tokenize(l):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            l_lemmatized_str.append(lemma)
            if w not in SW:
                l_lemmas_WO_SW.append(lemma)

    l_lemmatized_str = ' '.join(l_lemmatized_str)

    r_lemmas_WO_SW = []
    r_lemmatized_str = []
    for w in word_tokenize(r):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            r_lemmatized_str.append(lemma)
            if w not in SW:
                r_lemmas_WO_SW.append(lemma)

    r_lemmatized_str = ' '.join(r_lemmatized_str)

    # l_lemmatized_str = ' '.join([morph.parse(w.lower())[0].normal_form for w in word_tokenize(l) if w not in punct])
    # r_lemmatized_str = ' '.join([morph.parse(w.lower())[0].normal_form for w in word_tokenize(r) if w not in punct])

    # l_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(l) if w not in SW and w not in punct]
    # r_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(r) if w not in SW and w not in punct]

    is_hyponym_in_r = 0
    is_hyponym_in_l = 0

    for w in l_lemmas_WO_SW:
        if is_hyponym_in_r == 1:
            break
        lemmatized_hyponyms = [get_lemmatized_word_phrase(j) for i in get_all_hyponyms(w) for j in i.split(',')]
        for h in lemmatized_hyponyms:
            if h in r_lemmatized_str:
                print(f"Гипоним: {h} от слова: {w}")
                # print(l)
                # print(r)
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
                    # print(l)
                    # print(r)
                    is_hyponym_in_l = 1

    return is_hyponym_in_r | is_hyponym_in_l


# %%
def is_hypernym_in_sent_pair(sent_pair: list[list[str]]):
    l = sent_pair[0]
    r = sent_pair[1]

    l_lemmas_WO_SW = []
    l_lemmatized_str = []
    for w in word_tokenize(l):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            l_lemmatized_str.append(lemma)
            if w not in SW:
                l_lemmas_WO_SW.append(lemma)

    l_lemmatized_str = ' '.join(l_lemmatized_str)

    r_lemmas_WO_SW = []
    r_lemmatized_str = []
    for w in word_tokenize(r):
        if w not in punct:
            lemma = morph.parse(w.lower())[0].normal_form
            r_lemmatized_str.append(lemma)
            if w not in SW:
                r_lemmas_WO_SW.append(lemma)

    r_lemmatized_str = ' '.join(r_lemmatized_str)

    # l_lemmatized_str = ' '.join([morph.parse(w.lower())[0].normal_form for w in word_tokenize(l) if w not in punct])
    # r_lemmatized_str = ' '.join([morph.parse(w.lower())[0].normal_form for w in word_tokenize(r) if w not in punct])

    # l_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(l) if w not in SW and w not in punct]
    # r_lemmas_WO_SW = [morph.parse(w.lower())[0].normal_form for w in word_tokenize(r) if w not in SW and w not in punct]

    is_hypernym_in_r = 0
    is_hypernym_in_l = 0

    for w in l_lemmas_WO_SW:
        if is_hypernym_in_r == 1:
            break
        lemmatized_hypernyms = [get_lemmatized_word_phrase(j) for i in get_all_hypernyms(w) for j in i.split(',')]
        for h in lemmatized_hypernyms:
            if h in r_lemmatized_str:
                print(f"Гиперним: {h} от слова: {w}")
                # print(l)
                # print(r)
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
                    # print(l)
                    # print(r)
                    is_hypernym_in_l = 1

    return is_hypernym_in_r | is_hypernym_in_l


# %%
personal_pronouns = {'он', 'она', 'оно', 'они', 'его', 'ему', 'им', 'нём', 'нем', 'него', 'ее', 'её', 'ей', 'ею' 'ней',
                     'им',
                     'их', 'ими', 'них', 'собой', 'себе', 'себя'}


# %%
def is_anph_in_pair(sent_pair: list[list[str]]) -> int:
    l = sent_pair[0]
    r = sent_pair[1]

    l_tokenized = word_tokenize(l)
    r_tokenized = word_tokenize(r)

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
                # print(j)
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


def get_lemmatized_word_phrase(word_phrase: str):
    return ' '.join([morph.parse(i.lower().strip())[0].normal_form for i in word_phrase.split(' ')])


def process_sent_pair(pair):
    return [is_wordform_rep_in_pair(pair),
            is_anph_in_pair(pair),
            is_deriv_in_sent_pair(pair),
            is_hyponym_in_sent_pair(pair),
            is_hypernym_in_sent_pair(pair)]


if __name__ == '__main__':
    re_translation = "RU_[0-9]{1}_[0-9]{1,3}_[0-9]{1,3}.txt"
    files = sorted([i for i in os.listdir('rltc 2/')])
    rus_trans_texts = [open('rltc 2/' + f, 'r', errors="ignore").read() for f in files if
                       f != '.DS_Store' and re.match(re_translation, f)]

    # %%
    all_sents = []
    for i in rus_trans_texts:
        _ = sent_tokenize(i)
        for j in _:
            all_sents.append(j)
    # %%
    valid_sents = []
    lengts = []
    for s in all_sents:
        l = len([w for w in word_tokenize(s) if w not in punct])
        if l == 0 or l == 1:
            # print(s)
            pass
        else:
            valid_sents.append(s)
        lengts.append(l)
    # %%
    # nlp = spacy.load("ru_core_news_sm")
    # %%
    random.seed(42)
    valid_sents_shuffled = random.sample(valid_sents, len(valid_sents))
    # %%
    sent_pairs_shuffled = [[valid_sents_shuffled[i], valid_sents_shuffled[i + 1]] for i in
                           range(len(valid_sents_shuffled) - 1)]

    num_cores = multiprocessing.cpu_count()
    indices_parts = np.array_split(list(range(len(sent_pairs_shuffled))), 8)
    chunks = [sent_pairs_shuffled[indices_parts[i][0]:indices_parts[i][-1] + 1] for i in range(len(indices_parts))]

    # with multiprocessing.Pool(processes=num_cores) as pool:
    #     with tqdm(total=len(chunks[0])) as progress:
    #         results = []
    #
    #         for pair in chunks[0]:
    #             result = pool.apply_async(process_sent_pair, (pair, ))
    #             result.get()
    #             progress.update()
    #             results.append(result)
    #
    #         for result in results:
    #             print(result.get())

    # concurrent.futures.ThreadPoolExecutor
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(sent_pairs_shuffled)) as progress:
            futures = []

            for pair in sent_pairs_shuffled:
                future = pool.submit(process_sent_pair, pair)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)

    with open("processed_sent_pairs.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(results)
    # print(results)
