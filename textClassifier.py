
import random
import re
from datetime import datetime

import Stemmer
import emoji
from emoji import *
import numpy as np
import sklearn.model_selection
from sklearn.svm import SVC

from scipy.sparse import save_npz, dok_matrix, load_npz


def return_common_words(corr, incorr):
    doc_dict = load_text_file_as_dict('train.txt')
    corr_word_list = []
    corr_word_set = set()

    incorr_word_list = []
    incorr_word_set = set()

    for i in corr:
        doc = i[1]

        if doc_dict.get(str(doc)):
            bow = doc_dict.get(str(doc))[1]
            corr_word_set.update(bow)
            corr_word_list.extend(bow)


    for i in incorr:
        doc = i[1]

        if doc_dict.get(str(doc)):
            bow = doc_dict.get(str(doc))[1]
            incorr_word_set.update(bow)
            incorr_word_list.extend(bow)

    with open('dev.corr.wordlist', 'w') as f:
        ord_word_list = []

        for word in corr_word_set:
            number = len([word for wx in corr_word_list if wx == word])
            ord_word_list.append((word, number))

        ord_word_list.sort(key=lambda x:x[1])

        for word in ord_word_list:
            f.write(word[0] + ' : ' + str(word[1]) + '\n')

        f.close()

    with open('dev.incorr.wordlist', 'w') as f:
        ord_word_list = []

        for word in incorr_word_set:
            number = len([word for wx in incorr_word_list if wx == word])
            ord_word_list.append((word, number))

        ord_word_list.sort(key=lambda x:x[1])

        for word in ord_word_list:
            f.write(word[0] + ' : ' + str(word[1]) + '\n')

        f.close()

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
        recall.update({key : TP[key] / true_amount[key]})

    return recall


def get_precision(predicted_amount, TP):
    precision = {'-1': 0, '0': 0, '1': 0}

    for key in predicted_amount.keys():
        precision.update({key: TP[key] / predicted_amount[key]})

    return precision

def get_f_score(recall,precision):
    f_score = {'-1': 0, '0': 0, '1': 0}

    for key in precision.keys():
        prec = precision[key]
        reca = recall[key]
        f = 2*((reca*prec)/(reca+prec))
        f_score.update({key: f})

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

    # .py-score
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

def load_text_file(file_name, base_or_improved):

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
            if base_or_improved == 'base':
                bow = convert_to_wordlist(sw, text)
            if base_or_improved == 'improved':
                bow = convert_to_wordlist_improved(sw,text)

            entries.append([uid, sentiment, bow])

        f.close()

    return entries

def load_text_file_as_dict(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        entries = {}
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

            entries.update({uid : (sentiment, bow)})

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

def is_link(word_list):
    link_types = ['https//', 'http//']
    links = []
    for test_sub in link_types:
        links.extend(['/l' for i in re.finditer(test_sub, word_list)])
    return links

def get_pronouns(word_list):
    personal_pronouns = ['i', 'we', 'me', 'us', 'mine', 'ours', 'my', 'our']
    other_pronouns = ['he', 'his', 'him', 'she', 'her', 'hers', 'you', 'yours', 'they', 'theirs', 'them', 'it']
    p = ['/pp' for word in word_list if word in personal_pronouns]
    o = ['/op' for word in word_list if word in other_pronouns]
    p.extend(o)

    return p

def convert_to_wordlist_improved(stopWords, word_list):
    links = is_link(word_list)
    punctuation_list = ['/p' for p in word_list if not p.isalnum()]
    emojis = ['/e' for c in word_list if emoji.is_emoji(c)]
    word_list = tokenise(word_list)
    word_list = case_fold(word_list)

    pronouns = get_pronouns(word_list)

    word_list = sw_remove(word_list, stopWords)
    word_list = stemmer(word_list)
    word_list.extend(punctuation_list)
    word_list.extend(links)
    word_list.extend(emojis)
    word_list.extend(pronouns)

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


    def __load_npz(self, filename):
        self.matrix.update({filename: load_npz(filename).todok()})

    def __train_model(self, base_or_improved):
        X = self.matrix.get(base_or_improved + '.trainsm.npz')
        y = load_list(base_or_improved + '.train.keys')
        y = [int(i) for i in y]
        self.model = SVC(C=1000)
        self.model.fit(X,y)

    def dev(self, base_or_improved):
        self.__load_npz(base_or_improved + '.trainsm.npz')
        self.__load_npz(base_or_improved + '.devsm.npz')

        self.__train_model(base_or_improved)

        X = self.matrix.get(base_or_improved + '.trainsm.npz')
        y = load_list(base_or_improved + '.train.keys')

        train_score = self.model.score(X,y)

        X = self.matrix.get(base_or_improved + '.devsm.npz')
        y = load_list(base_or_improved + '.dev.keys')
        xy = self.model.predict(X)

        dev_score = self.model.score(X, y)
        recall, r_mac, precision, p_mac, f_score, f_mac = get_stats(y,xy)


        docs  = load_list(base_or_improved + '.dev.docs')

        with open(base_or_improved + '.dev.results', 'w') as f:

            f.write('Train score: \n')
            f.write(str(train_score) + '\n\n')
            f.write('Dev score: \n')
            f.write(str(dev_score) + '\n\n')

            f.write('Dev recall: \n')
            f.write('neg_recall: ' + str(recall.get('-1')) + '\n')
            f.write('neu_recall: ' + str(recall.get('0')) +'\n')
            f.write('pos_recall: ' + str(recall.get('1')) + '\n')
            f.write('mac_recall: ' + str(r_mac) +  '\n')

            f.write('Dev precision: \n')
            f.write('neg_precision: ' + str(precision.get('-1')) + '\n')
            f.write('neu_precision: ' + str(precision.get('0')) + '\n')
            f.write('pos_precision: ' + str(precision.get('1')) + '\n')
            f.write('mac_precision: ' + str(p_mac) + '\n')

            f.write('Dev f_score: \n')
            f.write('neg_f_score: ' + str(f_score.get('-1')) + '\n')
            f.write('neu_f_score: ' + str(f_score.get('0')) + '\n')
            f.write('pos_f_score: ' + str(f_score.get('1')) + '\n')
            f.write('mac_f-score: ' + str(f_mac) + '\n')



            class_dict = {'-1': 0, '0': 0, '1' : 0}
            corr = []
            incorr = []

            for i in range(len(y)):
                if y[i] == xy[i]:
                    corr.append((y[i],docs[i]))
                    n = class_dict.get(str(y[i]))
                    n += 1
                    class_dict[str(y[i])] = n

            f.write('\nCorrectly classified: \n\n')

            for i in class_dict.keys():
                f.write(i + ': ' + str(class_dict.get(i)) + '\n')

            class_dict = {'-1': 0, '0': 0, '1': 0}
            inc_dict_neg = {'0': 0, '1' : 0 }
            inc_dict_neu = {'-1': 0, '1': 0}
            inc_dict_pos = {'-1': 0, '0': 0}

            for i in range(len(y)):
                if y[i] != xy[i]:
                    incorr.append((y[i],docs[i]))
                    n = class_dict.get(str(y[i]))
                    n += 1
                    class_dict[str(y[i])] = n

                    if y[i] == -1:
                        n = inc_dict_neg.get(str(xy[i]))
                        n += 1
                        inc_dict_neg[str(xy[i])] = n

                    if y[i] == 0:
                        n = inc_dict_neu.get(str(xy[i]))
                        n += 1
                        inc_dict_neu[str(xy[i])] = n

                    if y[i] == 1:
                        n = inc_dict_pos.get(str(xy[i]))
                        n += 1
                        inc_dict_pos[str(xy[i])] = n

            corr.sort()
            incorr.sort()

            f.write('\nIncorrectly classified: \n\n')

            for i in class_dict.keys():
                f.write(i + ': ' + str(class_dict.get(i)) + '\n')

            f.write('\n Classified as: \n\n')

            for i in inc_dict_neg.keys():
                f.write('negative as ' + i + ': ' + str(inc_dict_neg.get(i)) + '\n')

            for i in inc_dict_neu.keys():
                f.write('neutral as ' + i + ': ' + str(inc_dict_neu.get(i)) + '\n')

            for i in inc_dict_pos.keys():
                f.write('positive as ' + i + ': ' + str(inc_dict_pos.get(i)) + '\n')

            f.close()

        with open(base_or_improved + '.dev.corr_docs', 'w') as f:

            for i in corr:
                f.write(str(i[0]) + ': ' + str(i[1]) + '\n')


            f.close()

        with open(base_or_improved + '.dev.incorr_docs', 'w') as f:

            for i in incorr:
                f.write(str(i[0]) + ': ' + str(i[1]) + '\n')

            f.close()

        return corr, incorr








    def test(self, base_or_improved):
        self.__load_npz(base_or_improved + '.trainsm.npz')
        self.__load_npz(base_or_improved + '.testsm.npz')

        self.__train_model(base_or_improved)

        X = self.matrix.get(base_or_improved + '.trainsm.npz')
        y = load_list(base_or_improved + '.train.keys')

        train_score = self.model.score(X, y)

        X = self.matrix.get(base_or_improved + '.testsm.npz')
        y = load_list(base_or_improved + '.test.keys')
        xy = self.model.predict(X)

        test_score = self.model.score(X, y)


    def file_load_label(self, filename, base_or_improved):
        entries = load_text_file(filename, base_or_improved)
        self.wuid = id_entries(entries)
        self.doc_id , self.class_id = get_classifier_doc(entries)

        return entries

    def files_to_matrix(self, base_or_improved):

        start_time = datetime.now()

        test = self.file_load_label('test.txt', base_or_improved)
        train = self.file_load_label('train.txt', base_or_improved)
        random.shuffle(test)
        random.shuffle(train)

        train, dev = sklearn.model_selection.train_test_split(train, train_size=0.9)
        word_list = get_word_set(train, self.wuid)
        save_list(word_list, 'all.words')


        self.__baseline_entries_to_sm(test, base_or_improved + '.test', word_list)
        self.__baseline_entries_to_sm(train, base_or_improved + '.train', word_list)
        self.__baseline_entries_to_sm(dev, base_or_improved + '.dev', word_list)

        print(datetime.now() - start_time)

    def files_to_matrix_improved(self):

        start_time = datetime.now()

        test = self.file_load_label('test.txt')
        train = self.file_load_label('train.txt')
        random.shuffle(test)
        random.shuffle(train)

        train, dev = sklearn.model_selection.train_test_split(train, train_size=0.9)
        word_list = get_word_set(train, self.wuid)
        save_list(word_list, 'all.words')


        self.__baseline_entries_to_sm(test, 'improved.test', word_list)
        self.__baseline_entries_to_sm(train, 'improved.train', word_list)
        self.__baseline_entries_to_sm(dev, 'improved.dev', word_list)

        print(datetime.now() - start_time)


    def __baseline_entries_to_sm(self,dataset, key, word_list):

        sm = []

        doc_list  = []  # list of docs in dataset
        class_list = [] # corresponding class of doc
        word_id_list = []

        class_id = {'negative': -1, 'neutral': 0, 'positive': 1}

        for doc in dataset:
            words = doc[2]
            doc_class = doc[1]
            doc_name = doc[0]



            doc_list.append(doc_name)
            class_list.append(class_id.get(doc_class))

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










