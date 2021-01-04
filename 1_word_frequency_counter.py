import os
import re
import nltk
import pandas as pd
import pickle as pkl
from copy import deepcopy
from nltk.corpus import stopwords
from gensim.models import ldamodel, Phrases, ldamulticore
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary


stop_words = stopwords.words('english')


def get_training_corpus(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder, file) for file in files]
    return files


def lemma(sentence):
    # lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    result = []
    for word in sentence:
        result.append(porter_stemmer.stem(word))
    return result


def load_company_name(filename):
    dataframe = pd.read_excel(filename)
    name = dataframe['Name'].tolist()
    return name


def counts(company_names, filenames, folder):
    dicts = {}
    dicts2 = {}
    contents = []
    for filename in filenames:
        basename = os.path.basename(filename)
        if basename[0] == '.':
            print(basename)
            continue
        with open(filename, 'r') as file:
            contents += file.readlines()

    for company_name in company_names:
        if not isinstance(company_name, str):
            continue
        company_name = company_name.lower()
        words = company_name.split()

        for word in words:
            if word not in dicts.keys():
                dicts.setdefault(word, 0)

    for content in contents:
        content = content.lower()
        words = nltk.word_tokenize(content)
        words = lemma(words)
        for word in words:
            if word not in dicts2:
                dicts2.setdefault(word, 0)
            dicts2[word] += 1

    for word in dicts:
        if word in dicts2.keys():
            dicts[word] = deepcopy(dicts2[word])
    dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)
    # print(dicts)
    with open(os.path.join(folder, 'company_word_freq.csv'), 'w', encoding='utf8') as file:
        for item in dicts:
            file.write(item[0] + ',' + str(item[1]) + '\n')
    dicts2 = sorted(dicts2.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(folder, 'all_words_freq.csv'), 'w', encoding='utf8') as file:
        for item in dicts2:
            file.write(item[0] + ',' + str(item[1]) + '\n')


if __name__ == "__main__":
    # need to modify
    folder = './2017'
    files = get_training_corpus(folder)
    company = load_company_name(
        './cik_ticker.xlsx')
    counts(company, files, folder)
