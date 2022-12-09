
import random
import re

import Stemmer
import numpy as np
import sklearn.model_selection
from sklearn.svm import SVC

from scipy.sparse import save_npz, dok_matrix, load_npz

def get_positives(pairs):
    TP = {'-1': 0, '0': 0, '1': 0}
    predicted_amount = {'-1': 0, '0': 0, '1': 0}
    true_amount = {'-1': 0, '0': 0, '1': 0}

    for pair in pairs:
        if pair[0] == pair[1]:
            l = TP.get(str(pair[0]))
            l += 1
            TP[str(pair[0])] = l

        # class that was predicted
        l = predicted_amount.get(str(pair[1]))
        l += 1
        predicted_amount[str(pair[1])] = l

        # true class
        l = true_amount.get(str(pair[0]))
        l += 1
        true_amount[str(pair[0])] = l

    return TP, predicted_amount, true_amount

def get_recall(true_amount, TP):
    recall = {'-1': 0, '0': 0, '1': 0}

    for key in true_amount.keys():
        recall[key].update({key : TP[key] / true_amount[key]})

    return recall


def get_precision(predicted_amount, TP):
    precision = {'-1': 0, '0': 0, '1': 0}

    for key in predicted_amount.keys():
        precision[key].update({key: TP[key] / predicted_amount[key]})

    return precision

def get_f_score(recall,precision):
    f_score = {'-1': 0, '0': 0, '1': 0}

    for key in precision.keys():
        prec = precision[key]
        reca = recall[key]
        f = 2*((reca*prec)/(reca+prec))
        f_score[key].update({key: f})

    return f_score

def get_macros(recall, precision, f_score):
    r_mac = sum([recall[key] for key in recall.keys()]) / 3
    p_mac = sum([precision[key] for key in precision.keys()]) / 3
    f_mac = sum([f_score[key] for key in f_score.keys()]) / 3

    return r_mac, p_mac, f_mac

def get_stats(trueY,predictY):

    pairs = list(zip(trueY, predictY))

    TP, predicted_amount, true_amount = get_positives(pairs)

    # recall - TP/ true_amount
    recall = get_recall(true_amount, TP)

    # precision - TP / predicted_amount
    precision = get_precision(predicted_amount, TP)

    # f-score
    f_score = get_f_score(recall,precision)


    r_mac, p_mac, f_mac = get_macros(recall,precision,f_score)

    return recall, r_mac, precision, p_mac, f_score, f_mac



def save_list(list, file_name):

    with open(file_name, 'w',encoding='utf-8') as f:
        q = file_name.split('.')
        f.write(q[1])
        f.write('\n')

        for i in list:
            f.write(str(i))
            f.write('\n')

        f.close()

def load_list(filename):

    list = []

    with open(filename, 'r', encoding='utf-8') as f:
        f.readline()
        read = f.read()
        read = read.split('\n')

        for i in read:
            if i == '' :
                continue
            list.append(int(i))

    return list

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

    return entries



def get_classifier_doc(entries):
    count = 0
    doc_id = {}
    class_id = {}
    classes = {'negative' : -1,  'neutral' : 0, 'positive' : 1}
    for doc in entries:
        count += 1
        y = doc[1]
        y_class = classes.get(y)
        doc_id.update({doc[0] : str(count)})
        class_id.update({str(count) : y_class})

    return doc_id, class_id

def id_entries(dataset):            # id' all entries
    print('hi')
    wuid_dict = {}
    word_id = set()
    word_counter = 0

    for i in range(len(dataset)): # through docs
        words = dataset[i][2]
        for word in words:  # through tweet
            if not (word in word_id):
                word_counter += 1
                word_id.add(word)
                print('GREAT SUCCESS')
                wuid_dict.update({word: str(word_counter)})

    return wuid_dict

def get_word_set(dataset, ids):
    word_ids = set()
    for doc in dataset:
        words = doc[2]
        for word in words:
            word_id = ids.get(word)
            word_ids.add(word_id)

    return word_ids

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
        self.doc_id = None
        self.class_id = None
        self.wuid = None
        self.matrix = {}


    def load_npz(self, filename):
        self.matrix.update({filename: load_npz(filename).todok()})

    def train_model(self):
        X = self.matrix.get('trainsm.npz')
        y = load_list('train.keys')
        y = [int(i) for i in y]
        self.model = SVC(C=1000)
        self.model.fit(X,y)

    def test(self):
        X = self.matrix.get('trainsm.npz')
        y = load_list('train.keys')
        xy = self.model.predict(X)

        for i in range(len(y)):
            w = str(y[i]) + ', ' + str(xy[i])
            print(w)
        print(self.model.score(X,y))

        X = self.matrix.get('testsm.npz')
        y = load_list('test.keys')
        print(self.model.score(X, y))

        return y

    def file_load_label(self, filename):
        entries = load_text_file(filename)
        self.wuid = id_entries(entries)
        self.doc_id , self.class_id = get_classifier_doc(entries)

        return entries

    def files_to_matrix(self,filename, base):

        test = self.file_load_label('test.txt')
        train = self.file_load_label('train.txt')
        random.shuffle(test)
        random.shuffle(train)

        train, dev = sklearn.model_selection.train_test_split(train, train_size=0.9)
        word_list = get_word_set(train, self.wuid)
        save_list(word_list, 'all.words')


        self.__baseline_entries_to_sm(test, 'base.test', word_list)
        self.__baseline_entries_to_sm(train, 'base.train', word_list)
        self.__baseline_entries_to_sm(dev, 'base.dev', word_list)


    def __baseline_entries_to_sm(self,dataset, key, word_list):

        sm = []

        doc_list  = []  # list of docs in dataset
        class_list = [] # corresponding class of doc
        word_id_list = []

        for doc in dataset:
            words = doc[2]
            doc_class = doc[1]
            doc_name = doc[0]

            class_list.append(doc_class)
            doc_list.append(doc_name)
            class_list.append(doc_class)

            doc_word_vector = []


            for word in word_list:
                doc_word_vector.append(len([word for word_id in words if word == self.wuid.get(word_id)]))
                word_id_list.append(self.wuid.get(word))

            sm.append(doc_word_vector)



        sm = np.asarray(sm)


        dm_matrix = dok_matrix(sm)
        save_npz(key + 'sm.npz', dm_matrix.tocoo())

        save_list(doc_list, key + '.docs')
        save_list(class_list, key + '.keys')










