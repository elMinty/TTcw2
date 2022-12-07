import Stemmer
import re
import pandas as pd
import math
from gensim.models import LdaModel
from gensim.test.utils import common_texts, datapath
from gensim.corpora.dictionary import Dictionary


def index_to_word_dict(f):
    prev_word = ""
    set_dict = {}

    for line in f:  # for line in file

        if line == "\n":  # ignore blank newline lines
            continue

        if line[0] != "\t":  # if next line is a word
            line = line.strip("\n")
            line = line.strip(":")

            if "_docs" in line:
                set_dict[line] = 0

            else:
                set_dict[line] = {"OT": [], "NT": [], "Quran": [], "OT_length": 0, "NT_length": 0, "Quran_length": 0}

            prev_word = line  # update word for locations

        if line[0] == "\t":  # if word document and locations

            line = line.strip("\t")
            line = line.strip("\n")
            line = line.strip(" ")

            if "_docs" in prev_word:
                set_dict[prev_word] = line
                continue

            line = line.split('-')
            doc = int(line[0])  # doc number is first in string
            book = str(line[1]).strip(" ")

            if book == "OT":
                get_set = set_dict[prev_word]
                doc_list = get_set["OT"]
                doc_list.append(doc)
                get_set.update({"OT": doc_list, "OT_length": len(doc_list)})
                set_dict[prev_word] = get_set
                continue

            if book == "NT":
                get_set = set_dict[prev_word]
                doc_list = get_set["NT"]
                doc_list.append(doc)
                get_set.update({"NT": doc_list, "NT_length": len(doc_list)})
                set_dict[prev_word] = get_set
                continue

            if book == "Quran":
                get_set = set_dict[prev_word]
                doc_list = get_set["Quran"]
                doc_list.append(doc)
                get_set.update({"Quran": doc_list, "Quran_length": len(doc_list)})
                set_dict[prev_word] = get_set
                continue

    return set_dict


"""
functions from cw1
"""


def invert_index(doc_dict):
    dictionary = doc_dict

    index = {}

    for key in dictionary:
        # key = str(key + 1)  # doc number
        word_list: list = dictionary[key]

        if not word_list:
            continue

        for i in range(len(word_list)):
            word = word_list[i]  # word for dictionary

            if has_word(index, word):

                doc_list = index.get(word)

                if key in doc_list:
                    continue
                else:

                    doc_list.append(key)

                    index.update({word: doc_list})

            else:
                index.update({word: [key]})

    return index


def has_word(index, word):
    if index.get(word) is None:
        return False
    else:
        return True


def has_doc(doc, word):
    if word.get(doc) is None:
        return False
    else:
        return True


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


def list_to_string(list):
    out = ""
    i = 0
    while i < len(list) - 1:
        out += str(list[i])
        out += ","
        i += 1
    out += str(list[-1])

    return out


def index_toText(index, f):
    for word in index:

        f.write(word + ":\n")
        word_dic = index.get(word)

        if type(word_dic) is int:
            f.write("\t" + str(word_dic) + "\n")
            continue

        for document in word_dic:
            f.write("\t" + document + "\n")

        f.write("\n")


"""
end functions from cw1
"""

"""

N = total number of documents belonging to the Corpara

n11 = total number of documents containing term T within class C
n10 = total number of documents containing term T outwith class C
n01 = total number of documents not containing term T within class C
n00 = total number of documents not containing term T outside of class C

"""


def mutual_information(N, n11, n10, n01, n00):
    n1 = n11 + n10
    n_1 = n01 + n11
    n0 = n01 + n00
    n_0 = n10 + n00

    result = 0

    if n11 != 0:
        part11 = (n11 / N) * math.log2((N * n11) / (n1 * n_1))
        result += part11
    if n01 != 0:
        part01 = (n01 / N) * math.log2((N * n01) / (n0 * n_1))
        result += part01
    if n10 != 0:
        part10 = (n10 / N) * math.log2((N * n10) / (n1 * n_0))
        result += part10
    if n00 != 0:
        part00 = (n00 / N) * math.log2((N * n00) / (n0 * n_0))
        result += part00

    return result


def chi_squared(n11, n10, n01, n00):
    first = sum([n11, n10, n01, n00])
    second = ((n11 * n00) - (n10 * n01)) ** 2

    third = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)

    if third == 0:
        return 0

    return (first * second) / third


class Text:

    def __init__(self):
        self.amount_docs = {}
        self.doc_map = None
        self.word_dict = None
        self.corpora_text = {}
        self.corpora_class = {}
        self.stop_words = self.get_stop_words()
        self.lda = None

        self.token_scores = {}

    def invert_tsv(self, file_name):
        df = self.__load_tsv(file_name)
        self.__doc_to_index(df)

    def lda_tsv(self,file_name):
        df = self.__load_tsv(file_name)
        book_list = ['OT', 'NT', 'Quran']

        doc_list = []

        for book in book_list:
            df_book = df.loc[df['Book'] == book]
            sent_list = df_book['Sentence'].tolist()

            for i in range(len(sent_list)):
                word_list = convert_to_wordlist(self.stop_words, sent_list[i])
                doc_list.append(word_list)

        common_dictionary = Dictionary(doc_list)
        common_corpus = [common_dictionary.doc2bow(text) for text in doc_list]
        self.lda = LdaModel(common_corpus, id2word=common_dictionary,num_topics=20)
        temp_file = datapath("model")
        self.lda.save(temp_file)

        topics = [self.lda.get_document_topics(common_dictionary.doc2bow(text)) for text in doc_list]
        print(topics)

    def lda_get_scores(self):
        self.lda = LdaModel.load(datapath("model"))


    @staticmethod
    def __load_tsv(file_name):
        df = pd.DataFrame(columns=['Book', 'Sentence'])

        with open("train_and_dev.tsv") as file:
            for line in file:
                l = line.split('\t')
                df = df.append({'Book': l[0], 'Sentence': l[1]}, ignore_index=True)

        return df

    def __doc_to_index(self, df):
        book_list = ['OT', 'NT', 'Quran']
        overall = 0
        doc_list = {'OT_docs': 0, 'NT_docs': 0, 'Quran_docs': 0}

        for book in book_list:
            df_book = df.loc[df['Book'] == book]
            sent_list = df_book['Sentence'].tolist()
            docs = 0

            for i in range(len(sent_list)):
                word_list = convert_to_wordlist(self.stop_words, sent_list[i])
                self.corpora_text.update({str(i + 1 + overall) + " - " + book: word_list})
                overall += 1
                docs += 1

            book_docs = book + "_docs"
            doc_list.update({book_docs: docs})

        text_doc = invert_index(self.corpora_text)
        text_doc.update(doc_list)
        text_doc.update({'overall_docs': overall})

        with open('corpora.index', 'w') as f:
            index_toText(text_doc, f)
            f.close()

        with open("doc_map.corpora.index", 'w') as f:
            index_toText(self.corpora_class, f)
            f.close()

    def download_index(self, file_name):

        with open(file_name, 'r') as f:
            word_dict = index_to_word_dict(f)
            self.word_dict = word_dict
            f.close()

        self.amount_docs.update({'Overall': self.word_dict['overall_docs']})
        self.amount_docs.update({'OT': self.word_dict['OT_docs']})
        self.amount_docs.update({'NT': self.word_dict['NT_docs']})
        self.amount_docs.update({'Quran': self.word_dict['Quran_docs']})

        self.word_dict.pop('overall_docs')
        self.word_dict.pop('OT_docs')
        self.word_dict.pop('NT_docs')
        self.word_dict.pop('Quran_docs')

    def process(self):

        for word in self.word_dict:

            word_SCORES = {}

            get_set = self.word_dict[word]
            corpus_list = ['OT', 'NT', 'Quran']
            N = int(self.amount_docs['Overall'])

            corp_stats = {'OT': None, 'NT': None, 'Quran': None}
            stats_lists = []

            for corp in corpus_list:
                words_in = get_set[corp + '_length']
                words_out = int(self.amount_docs[corp]) - int(words_in)

                stats_lists.append([words_in, words_out])

            for corp in range(len(corpus_list)):
                n11 = int(stats_lists[corp][0])
                n01 = int(stats_lists[corp][1])
                n10 = int(stats_lists[corp - 1][0]) + int(stats_lists[corp - 2][0])
                n00 = int(stats_lists[corp - 1][1]) + int(stats_lists[corp - 2][1])
                mi = mutual_information(N, n11, n10, n01, n00)
                chi = chi_squared(n11, n10, n01, n00)

                word_SCORES.update({corpus_list[corp]: [mi, chi]})

            self.token_scores.update({word: word_SCORES})

    def return_top_ten(self):
        corpus_list = ['OT', 'NT', 'Quran']

        OT_mi = []
        NT_mi = []
        Quran_mi = []

        OT_chi = []
        NT_chi = []
        Quran_chi = []

        for word in self.token_scores:
            get_set = self.token_scores[word]

            get_set_OT = get_set['OT']
            OT_mi.append((get_set_OT[0], word))
            OT_chi.append((get_set_OT[1], word))

            get_set_NT = get_set['NT']
            NT_mi.append((get_set_NT[0], word))
            NT_chi.append((get_set_NT[1], word))

            get_set_Quran = get_set['Quran']
            Quran_mi.append((get_set_Quran[0], word))
            Quran_chi.append((get_set_Quran[1], word))

        OT_mi.sort(reverse=True)
        OT_chi.sort(reverse=True)

        NT_mi.sort(reverse=True)
        NT_chi.sort(reverse=True)

        Quran_mi.sort(reverse=True)
        Quran_chi.sort(reverse=True)

        stats = [[OT_mi, NT_mi, Quran_mi], [OT_chi, NT_chi, Quran_chi]]

        with open('index.stats', 'w') as f:
            scores = ['Mutual Information', 'CHI squared']
            corpus = ['OT', 'NT', 'Quran']

            for i in range(len(scores)):
                f.write(scores[i] + ':\n\n')
                for j in range(len(corpus)):
                    f.write(corpus[j] + '\n\n')
                    for k in range(20):
                        f.write(stats[i][j][k][1] + ', ' + str(stats[i][j][k][0]) + '\n')

            f.close()

    @staticmethod
    def get_stop_words():
        file = open("englishST.txt")
        stop_words = file.read()
        stop_words = re.split("\n", stop_words)
        file.close()

        return stop_words
