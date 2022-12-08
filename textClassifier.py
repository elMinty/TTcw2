
import random
import re

import Stemmer
import numpy as np
import sklearn.model_selection

from scipy.sparse import save_npz, dok_matrix, load_npz


def load_text_file(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        entries = []
        first_line = True

        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.split('\t')
            uid = line[0]
            sentiment = line[1]
            text = line[2]
            sw = get_stop_words()
            bow = convert_to_wordlist(sw, text)

            entries.append([uid, sentiment, bow])

        f.close()

    random.shuffle(entries)

    train, test = sklearn.model_selection.train_test_split(entries, train_size=0.8)

    return train,test

def get_classifier(entries):
    doc_class = []
    for doc in entries:
        doc_class.append(entries[1])

    dcuid_map = []
    count = 0
    dcuid_dic = {}
    for dc in doc_class:
        continue


def tokenise(words):
    word_list = (re.split("[^a-zA-Z]", words))
    word_list = list(filter(bool, word_list))

    return word_list


def case_fold(word_list):
    word_list = [i.lower() for i in word_list]

    return word_list


def sw_remove(word_list, stopWords):
    for i in stopWords:
        word_list = [j for j in word_list if j != i]

    return word_list


def stemmer(word_list):
    stem = Stemmer.Stemmer('english')
    word_list = [stem.stemWord(i) for i in word_list]

    return word_list


def convert_to_wordlist(stopWords, word_list):
    word_list = tokenise(word_list)
    word_list = case_fold(word_list)
    word_list = sw_remove(word_list, stopWords)
    word_list = stemmer(word_list)

    return word_list


def get_stop_words():
    file = open("englishST.txt")
    stop_words = file.read()
    stop_words = re.split("\n", stop_words)
    file.close()

    return stop_words


class Classify:

    def __init__(self):
        self.uid_dict = {'train':{},'test':{}}
        self.train, self.test = load_text_file('train.txt')
        self.matrix = {}


    def load_npz(self, filename):
        self.matrix.update({filename: load_npz(filename)})

    def file_to_matrix(self):
        self.__id_entries(self.train, 'train')
        self.__entries_to_sm(self.train, 'train')

        self.__id_entries(self.test, 'test')
        self.__entries_to_sm(self.test, 'test')


    def __entries_to_sm(self,dataset, key):

        sm = []

        word_map = self.uid_dict.get(key).get('word_map')
        wid_map = self.uid_dict.get(key).get('wid_map')
        for i in range(len(word_map)):
            word = word_map[i]
            uid = wid_map[i]
            word_vector = []

            for doc in dataset:
                d_occ_word = len([word for d_word in doc[2] if d_word == word])
                word_vector.append(d_occ_word)

            sm.append(word_vector)

        sm = np.asarray(sm)

        dm_matrix = dok_matrix(sm)
        save_npz(key + 'sm.npz', dm_matrix.tocoo())

    def __id_entries(self, dataset, key):            # id' all entries

        wid_map = []
        word_map = []
        word_id = set()
        word_counter = 0

        for i in range(len(dataset)): # through docs
            words = dataset[i][2]
            for word in words:  # through tweet
                if not (word in word_id):
                    word_counter += 1
                    word_id.add(word)
                    wid_map.append(word_counter)
                    word_map.append(word)

        dict_k = {'wid_map': wid_map, 'word_map': word_map}
        self.uid_dict.update({key:dict_k})






