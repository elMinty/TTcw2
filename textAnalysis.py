
import Stemmer
import re
import pandas as pd
import math

# def index_parse(f, book_dict):
#     for line in f:  # for line in file
#
#         if line == "\n":  # ignore blank newline lines
#             continue
#
#         if line[0] != "\t":  # if next line is a word
#             line = line.strip("\n")
#             line = line.strip(":")
#
#             pos_dict[line] = {}  # enter new key into dictionary
#
#             set_dict[line] = set()
#
#             prev_word = line  # update word for locations
#
#         if line[0] == "\t":  # if word document and locations

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
n00 = total number of documents 

"""


def mutual_information(N, n11, n10, n01, n00):
    n1 = n11 + n10
    n_1 = n01 + n11
    n0 = n01 + n00
    n_0 = n10 + n00

    first_part = (n11 / N) * math.log2((N * n11) / (n1 * n_1))

    second_part = (n01 / N) * math.log2((N * n01) / (n0 * n_1))

    third_part = (n10 / N) * math.log2((N * n10) / (n1 * n_0))

    fourth_part = (n00 / N) * math.log2((N * n00) / (n0 * n_0))

    result = first_part + second_part + third_part + fourth_part

    return result


class Text:

    def __init__(self):
        self.corpora_text = {}
        self.corpora_class = {}
        self.stop_words = self.get_stop_words()


    # def part2(self):
    #     """
    #     first load_tsv_to_index the tsv file
    #     with this have a file represented as an index and go from there to get results:
    #     :return: index to file
    #     """
    #
    #     self.index_load(self, file_name)

    def load_tsv_to_index(self, file_name):
        df = pd.DataFrame(columns=['Book', 'Sentence'])

        with open("train_and_dev.tsv") as file:

            for line in file:
                l = line.split('\t')
                df = df.append({'Book': l[0], 'Sentence': l[1]}, ignore_index=True)

        book_list = ['OT', 'NT', 'Quran']
        overall = 0

        for book in book_list:
            df_book = df.loc[df['Book'] == book]
            sent_list = df_book['Sentence'].tolist()
            doc_list = []

            for i in range(len(sent_list)):
                word_list = convert_to_wordlist(self.stop_words, sent_list[i])
                self.corpora_text.update({str(i+1+overall): word_list})
                doc_list.append(str(i+1+overall))
                self.corpora_class.update({str(i+1+overall): [book]})

            overall += len(doc_list)

        text_doc = invert_index(self.corpora_text)

        with open(file_name,'w') as f:
            index_toText(text_doc, f)
            f.close()

        with open("doc_map." + file_name, 'w') as f:
            index_toText(self.corpora_class, f)
            f.close()

    # def index_load(self,file_name):
    #     with open(file_name, 'r') as f:
    #         index_parse(f)
    #         f.close()

    @staticmethod
    def get_stop_words():
        file = open("englishST.txt")
        stop_words = file.read()
        stop_words = re.split("\n", stop_words)
        file.close()

        return stop_words

    # n00
    # n01
    # n10
    # n11
    # n - all instances in the table sum(n00,n01,...)

