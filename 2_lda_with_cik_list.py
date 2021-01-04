import os
import re
import nltk
import _thread
import threading
import pandas as pd
import pickle as pkl
from nltk.corpus import stopwords
from gensim.models import ldamodel, Phrases, ldamulticore
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary

stop_words = None
contents = []
failures = []
analysed_filenames = []


def read_cik_from_cik_file(filename):
    if not filename:
        return None
    return pd.read_excel(filename)


def load_stopwords():
    with open('./stopwords.txt', 'r') as file:
        stopwords = file.readlines()
        stopwords = [word.strip('\n') for word in stopwords]
    return stopwords


def lemma(sentence):
    # lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    result = []
    for word in sentence:
        result.append(porter_stemmer.stem(word))
    return result


def read_key_words(filename=None):
    dataframe = pd.read_excel(filename, sheet_name='key words')
    key_words = dataframe['Selected key words'].tolist()
    result = []
    for key_word in key_words:
        key_word = key_word.lower()
        key_word = key_word.split(';')
        key_word = [word.strip() for word in key_word]
        result += key_word
    return result


def word_tokenize(sentence):
    return nltk.word_tokenize(sentence)


def get_patterns(key_words):
    re_key_words = []
    key_words = [word_tokenize(words) for words in key_words]
    key_words = [lemma(words) for words in key_words]
    word_pattern = r'[\s\w+]{0,2}\s'
    for words in key_words:
        if len(words) == 1:
            re_key_words.append(re.compile(words[0]))
        else:
            pattern = word_pattern.join(words)
            re_key_words.append(re.compile(pattern))
    re_key_words.append(re.compile(r'co-\w+'))
    return re_key_words


def get_training_corpus(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder, file) for file in files]
    return files


def save_result(result, temp=False):
    filename = 'lda_result_for_all.tsv' if not temp else 'temp_lda_result_for_all.tsv'
    with open(filename, 'w', encoding='utf8') as file:
        for key in result:
            topics = result[key]
            file.write(key+'\n')
            for id, topic in enumerate(topics):
                content = 'topics_%d' % id + '\t' + '\t'.join(topic)
                file.write(content+'\n')


def pre_processing(line, patterns):
    line = line.split('\t')[0]
    line = line.replace('\"', ' ')
    line = line.replace('"', ' ')
    line = word_tokenize(line)
    line = [x for x in line if len(x) > 3]
    line = lemma(line)
    line = remove_stopwords(line)
    line = ' '.join(line)
    for pattern in patterns:
        if len(pattern.findall(line)) != 0:
            for item in pattern.findall(line):
                line = line.replace(item, item.replace(' ', '_'))
    return line


class Reader(threading.Thread):
    def __init__(self, filenames, patterns):
        threading.Thread.__init__(self)
        self.filenames = filenames
        self.patterns = patterns

    def run(self):
        # global contents
        for filename in self.filenames:
            print('read {}'.format(filename))
            try:
                content = self.get_content(filename, self.patterns)
                if content is not None:
                    contents.append(content)
                    analysed_filenames.append(filename)
            except:
                failures.append(filename)

    def get_content(self, filename, patterns):
        basename = os.path.basename(filename)
        if basename[0] == '.':
            return None
        with open(filename, 'r', encoding='utf8') as file:
            result = []
            result = file.readlines()
            if len(result) < 5:
                return None
            result = [pre_processing(line, patterns) for line in result]
            return ' '.join(result)


def tokenization(sentence):
    words_list = nltk.word_tokenize(sentence)
    word_pattern = re.compile('[a-zA-Z]+')
    result = []
    for word in words_list:
        if len(word_pattern.findall(word)) != 0:
            result.append(word)
    return result


def remove_stopwords(words_list):
    result = []
    useless_token = [':', ',', '.', '?', '!',
                     ':', '\'', '\"', '(', ')', '[', ']']
    for word in words_list:
        if word in stop_words or word in useless_token:
            continue
        result.append(word)
    return result


def topicWords(data):
    data = data.lstrip('(').strip(')')
    data = data.split('+')
    data = [dat.split('*')[1] for dat in data]
    data = [dat.replace('"', '') for dat in data]
    return data


def get_bigrams(data):
    bigram = Phrases(data, min_count=5)
    # result = []
    for idx in range(len(data)):
        for token in bigram[data[idx]]:
            if '_' in token:
                data[idx].append(token)


def get_top_content(top, content, words_list):
    result = []
    for word in content:
        if word in words_list:
            result.append(word)
    return result


def filter_frequency_under(data, frequency_map, threshold=10, need_to_remove_words=None):
    result = []
    for word in data:
        if word in need_to_remove_words:
            continue
        result.append(word)
    return result


def get_top(data, frequency_map, threshold=4000):
    need_to_leave_up = list(frequency_map.keys())[:threshold]
    result = []
    for word in data:
        if word in need_to_leave_up:
            result.append(word)
    return result


def lda_analysis(filenames, word_frequences=None):
    result = {}
    global contents
    global analysed_filenames

    def get_company(filename):
        name = os.path.basename(filename)
        name = name.split('_')
        if len(name) == 3:
            name = name[1]
        elif len(name) == 2:
            name = name[0]
        return name
    keywords = read_key_words(
        './Corpus_Building_OI_Practices.xlsx')
    patterns = get_patterns(keywords)
    number = len(filenames)
    cache = 8
    start = 0
    for thread_id in range(cache):
        end = (thread_id + 1) * (number // cache + 1)
        filenames_tmp = filenames[start: end]
        start = end
        try:
            thread = Reader(filenames_tmp, patterns)
            thread.start()
            thread.join()
            # _thread.start_new_thread(get_cache_content,(filenames_tmp, patterns))
        except:
            print('Thread {} cannot be started'.format(thread_id))
    print('==== read completely ====')
    with open('analysed_filenames.pkl', 'wb') as file:
        pkl.dump(analysed_filenames, file)
    with open('analysed_filenames.txt', 'w') as file:
        analysed_filenames = [os.path.basename(
            item) for item in analysed_filenames]
        file.write('\n'.join(analysed_filenames))
    print('==== tokenization ====')
    contents = [tokenization(c) for c in contents]
    print('==== filter by frequency ====')
    need_to_remove_words = [
        word for word in word_frequences if word_frequences[word] < 50]
    contents = [filter_frequency_under(
        content, word_frequences, need_to_remove_words=need_to_remove_words) for content in contents]
    # print('==== get top N words ====')
    # contents = [get_top(content, word_frequences) for content in contents]
    # get_bigrams(contents)
    print('==== Begin LDA analysis. ====')
    dictionary = Dictionary(contents)
    # dictionary.filter_extremes(no_below=10)
    topic_nums = 50
    corpus = [dictionary.doc2bow(text) for text in contents]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of unique documents: %d' % len(corpus))
    models = ldamulticore.LdaMulticore(corpus=corpus, num_topics=topic_nums, id2word=dictionary,
                                       chunksize=3000, iterations=100, passes=100, workers=8, random_state=50)
    models.save('passes_100_chunsize_8000.lda')
    topics = models.print_topics(num_topics=topic_nums)
    item = []
    for i in topics:
        try:
            item.append(topicWords(i[1]))
        except:
            item.append([])
    result.setdefault('all company result', item)
    doc_topics = models.get_document_topics(bow=corpus)
    return result, doc_topics


def save_doc_topics(doc_topics):
    with open('doc_topics_passes_100.lda', 'wb') as file:
        pkl.dump(doc_topics, file)


def load_word_frequency(filename):
    results = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(',')
            line = line.split(',')

            if len(line) == 3 or len(line) == 1:
                line = [',', line[-1]]
            if line[0] in stop_words:
                continue
            results.setdefault(line[0], int(line[1]))
        return results


def remove_useless_files(files, cik_list=None):
    result = []
    if cik_list is None:
        return files
    for file in files:
        cik_name = file.split('_')[0]
        if cik_name not in cik_list:
            result.append(file)
    return result


if __name__ == "__main__":
    folder = './2017'
    word_frequency_file = './all_words_freq.csv'
    print('==== load cik list ====')
    cik_filename = None  # TODO: to modify the filename
    cik_list = read_cik_from_cik_file(cik_filename)
    print('==== load stopwords ====')
    stop_words = load_stopwords()
    print('==== lemma stopwords ====')
    stop_words = lemma(stop_words)
    print('==== load wordfrequency ====')
    word_frequency = load_word_frequency(word_frequency_file)
    files = get_training_corpus(folder)
    print('==== remove useless filenames ====')
    files = remove_useless_files(files, cik_list)
    result, doc_topics = lda_analysis(files, word_frequency)
    save_result(result)
    save_doc_topics(doc_topics)
    print(doc_topics)
