from datetime import datetime
from pydoc import doc
import Stemmer
import re
import xml.etree.ElementTree as ET
import pandas as pd


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


def convert_to_wordlist(stopWords, word_list, Q):
    if not Q:
        word_list = tokenise(word_list)

    word_list = case_fold(word_list)
    word_list = sw_remove(word_list, stopWords)
    word_list = stemmer(word_list)

    return word_list


class Text:

    def __init__(self):
        self.df = None

        self.part2()

    def part2(self):
        self.__load_tsv()

    def __load_tsv(self):
        self.df = pd.read_csv('train_and_dev.tsv', sep='\t')
        print(self.df)


if __name__ == "__main__":
    t = Text()
    exit()
