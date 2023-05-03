import argparse
from pathlib import Path
import glob
import os
import re
import math
from typing import Dict, List, Tuple
from collections import Counter, OrderedDict

import sys
import numpy as np
import pandas as pd


def get_top_k(kv_dict: Dict[str, float], k: int = 20) -> List[Tuple[str, float]]:
    """
    Returns the top 'k' key-value pairs from a dictionary, based on their values.

    :param kv_dict: A dictionary of key-value pairs, where the values are scores or counts.
    :param k: The number of key-value pairs with top 'k' values (default k=20).
    :return: A list of the top 'k' key-value pairs from the dictionary, sorted by value.

    Example:
    # >>> kv_dict = {'apple': 5.0, 'banana': 3.0, 'orange': 2.5, 'peach': 1.0}
    # >>> get_top_k(kv_dict, 2)
    [('apple', 5.0), ('banana', 3.0)]
    """
    # Sort the dictionary by value and return the top 'k' key-value pairs
    # sort high to low on value's floating point number
    kv_sorted = sorted(kv_dict.items(), key=lambda x:x[1], reverse=True)
    top_k = kv_sorted[:k] # get slice of first k items

    return top_k


def sort_dictionary_by_value(dict_in: Dict[str, float], direction: str = "descending") -> List[Tuple[str, float]]:
    """
    Sort a dictionary of key-value pairs by their values.

    :param dict_in: A dictionary of key-value pairs, where the values are scores or counts.
    :param direction: The sorting direction, either 'descending' (default) or 'ascending'.
    :return: A list of the key-value pairs from the dictionary, sorted by value.

    Example:
    # >>> kv_dict = {'apple': 5.0, 'banana': 3.0, 'orange': 2.5, 'peach': 1.0}
    # >>> sort_dictionary_by_value(kv_dict)
    [('peach', 1.0), ('orange', 2.5), ('banana', 3.0), ('apple', 5.0)]
    """
    sort_dict = []  # TODO: fix me
    # Sort the dictionary  dict_in by value
    # Reverse the order if the direction is 'descending'

    out = Counter(dict_in)
    if direction == "ascending":
        sort_dict = list(out.most_common()[::-1])
    else: # default
        sort_dict = list(out.most_common())

    return sort_dict


def strip_non_ascii(string):
    """Returns the string without non ASCII characters"""
    stripped = (c for c in string if 0 < ord(c) < 127)
    return "".join(stripped)


def clean_text(s):
    """Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana'"""
    s = strip_non_ascii(s)
    s = re.sub("[^a-z A-Z]", "", s)
    s = s.replace(" n ", " ")

    return s


def clean_corpus(corpus):
    """Run clean_text() on each sonnet in the corpus

    :param corpus:  corpus dict with keys set as filenames and contents as a single string of the respective sonnet.
    :type corpus:   dict

    :return     corpus with text cleaned and tokenized. Still a dictionary with keys being file names, but contents
                now the cleaned, tokenized content.
    """
    for key in corpus.keys():
        # clean each exemplar (i.e., sonnet) in corpus

        # call function provided to clean text of all non-alphabetical characters and tokenize by " " via split()
        corpus[key] = clean_text(corpus[key]).split()

    return corpus


def read_sonnets(fin):
    """
    Passes image through network, returning output of specified layer.

    :param fin: fin can be a directory path containing TXT files to process or to a single file,

    :return: (dict) Contents of sonnets with filename (i.e., sonnet ID) as the keys and cleaned text as the values.
    """

    """ reads and cleans list of text files, which are sonnets in this assignment"""

    if Path(fin).is_file():
        f_sonnets = [fin]
    elif Path(fin).is_dir():
        f_sonnets = glob.glob(fin + os.sep + "*.txt")
    else:
        print("Filepath of sonnet not found!")
        return None

    if len(f_sonnets) > 1:
        text_names = []
        for entry in f_sonnets:
            text_names.append(entry.split('/')[-1].split('.txt')[0])
        # print(f'{text_names = }')

        sorted_text_names = [ int(x) for x in text_names ]
        sorted_text_names.sort()

        f_sonnets = [ f'{fin}{x}.txt' for x in sorted_text_names ] # sort 1...154 numerically
        # from natsort import natsorted # ModuleNotFoundError: No module named 'natsort'
        # f_sonnets = natsorted(f_sonnets)

    sonnets = {}

    for f in f_sonnets:
        sonnet_id = Path(f).stem
        data = []
        with open(f, "r") as file:
            data.append(file.readline().replace("\\n", "").replace("\\r", ""))

        sonnets[sonnet_id] = clean_text("".join(data))

    # sort by key numerically # make all words lower-case
    sonnets = OrderedDict({str(int(k)) : v.lower() for k, v in sonnets.items()})
    # print(f'{sonnets = }')

    return sonnets


def tf(document: List[str]) -> Dict[str, int]:
    """
    Calculate the term frequency (TF) for each word in a document.

    The term frequency of a word is defined as the number of times it appears in the document.

    :param document: A list of words representing the document.
    :return: A dictionary where the keys are words and the values are their term frequency in the document.

    Example:
    # >>> doc = ['apple', 'banana', 'orange', 'peach', 'apple']
    # >>> tf(doc)
    {'apple': 2, 'banana': 1, 'orange': 1, 'peach': 1}
    """
    # Count the occurrences of each word in the document
    # document_tf = {}  # TODO: fix me
    document_tf = Counter(document)

    document_tf = OrderedDict(document_tf.most_common())

    return document_tf


def idf(corpus: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate the inverted document frequency (IDF) for each word in a corpus.

    The IDF of a word is defined as log(N/df), where N is the total number of documents in the corpus and df
    is the number of documents that contain the word.

    :param corpus: A dictionary of documents, where each document is represented as a list of words.
    :return: A dictionary where the keys are words and the values are their IDF scores.

    Example:
    # >>> corpus = {"doc1": ["apple", "banana", "orange"], "doc2": ["banana", "peach"], "doc3": ["orange", "peach"]}
    # >>> idf(corpus)
    {'apple': 1.0986122886681098, 'banana': 0.4054651081081644, 'orange': 0.4054651081081644, 'peach': 0.4054651081081644}
    """

    all_words = [word for sonnet in corpus.values() for word in sonnet] # all words together
    uniq_all_words = sorted(list(set(list(all_words)))) # unique words
    # print(f'{all_words = }');print(f'{uniq_all_words = }')
    # print(len(all_words)); print(len(uniq_all_words))

    # Calculate the IDF for each word

    # key = word; value = doc_count
    # doc count is number of documents the word occurs in
    corpus_idf = OrderedDict.fromkeys(uniq_all_words, 0) # TODO: FIX ME

    for sonnet_no, sonnet_words in corpus.items():

        uniq_sonnet_words = list(set(sonnet_words)) # only need want to hit once per document

        for word in uniq_sonnet_words:

            doc_count = corpus_idf[word] + 1
            corpus_idf.update({word : doc_count}) # add 1 for every document hitting word

    N = len(corpus.keys())
    for sonnet_word, df in corpus_idf.items():

        new_idf = math.log(N / float(df))

        corpus_idf[sonnet_word] = float(new_idf) # change value from df to idf score

    return corpus_idf  # word : idf_score


def tf_idf(corpus_idf: Dict[str, float], sonnet_tf: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the TF-IDF scores for each word in a sonnet, using a pre-computed IDF dictionary.

    The TF-IDF score of a word is defined as tf(word) * idf(word), where tf(word) is the term frequency of the word
    in the sonnet and idf(word) is the inverse document frequency of the word in the corpus.

    :param corpus_idf: A dictionary where the keys are words and the values are their IDF scores.
    :param sonnet_tf: A dictionary where the keys are words and the values are their TF scores in the sonnet.
    :return: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.

    Example:
    # >>> corpus_idf = {'apple': 1.0986122886681098, 'banana': 0.4054651081081644, 'orange': 0.4054651081081644, 'peach': 0.6931471805599453}
    # >>> sonnet_tf = {'apple': 2, 'banana': 1, 'orange': 0, 'peach': 3}
    # >>> tf_idf(corpus_idf, sonnet_tf)
    {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0, 'peach': 2.0794415416798357}
    """

    corpus_tf_idf = {}  # TODO: fix me

    corpus_tf_idf.fromkeys(corpus_idf, 0)

    for sonnet_word, sonnet_val in sonnet_tf.items():
        tf = sonnet_val
        idf = 0.0 # in case it does not exist in corpus words (testcases)
        if sonnet_word in corpus_idf.keys():
            idf = corpus_idf[sonnet_word]
        tf_idf = float(tf * idf)
        corpus_tf_idf[sonnet_word] = tf_idf

    return corpus_tf_idf


def cosine_sim(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Calculate the cosine similarity between two tf-idf vectors.

    :param vec1: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :param vec2: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :return: The cosine of the vectors

    Example:
    # >>> vec1 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0}
    # >>> vec2 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'peach': 2.0794415416798357}
    # >>> cosine_sim(vec1, vec2)
    # >>> 0.7320230293693564

    """
    similarity = 0  # TODO: fix me
    # loop through all vec1, vec2 values and sum up...then square...then mult...then sqrt
    vec1_tfidf_sum = 0; vec2_tfidf_sum = 0

    # 1 = {A, B, D}
    # 2 = {A, C, E}
    numerator = 0

    for v1, v2 in zip(vec1.values(), vec2.values()):
        vec1_tfidf_sum += math.pow(v1, 2)
        vec2_tfidf_sum += math.pow(v2, 2)

        numerator += v1 * v2

    denominator = math.sqrt(vec1_tfidf_sum * vec2_tfidf_sum)

    similarity = numerator / denominator

    return similarity


def euclidean_distance():
    ...


def manhattan_distance():
    ...


def bm25():
    ...


def sbert():
    ...



if __name__ == "__main__":

    # python3 text_analyzer.py - i "./data/shakespeare_sonnets/3.txt"

    parser = argparse.ArgumentParser(description="Text Analysis through TFIDF computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("-i", "--input", type=str,
        default="./data/shakespeare_sonnets/1.txt",
        help="Input text file or files.",)

    parser.add_argument("-c", "--corpus", type=str,
        default="./data/shakespeare_sonnets/",
        help="Directory containing document collection (i.e., corpus)",)

    parser.add_argument("--tfidf",
        help="Determine the TF IDF of a document w.r.t. a given corpus",
        action="store_true",)

    args = parser.parse_args()

    # return dictionary with keys corresponding to file names and values being the respective contents
    corpus = read_sonnets(args.corpus)

    # return corpus (dict) with each sonnet cleaned and tokenized for further processing
    corpus = clean_corpus(corpus)

    # assign 1.txt to variable sonnet to process and find its TF (Note corpus is of type dic, but sonnet1 is just a str)
    sonnet1 = corpus["1"]

    # input = read_sonnets(args.input)
    if args.input != "./data/shakespeare_sonnets/1.txt":
        sonnet_num = args.input.split('/')[-1].split('.')[0] # get sonnet num (2...154)
        sonnet1 = corpus[sonnet_num]
        print(f'{sonnet1 = }')

    # determine tf of sonnet
    sonnet1_tf = tf(sonnet1)
    print(f'{sonnet1_tf = }')

    # get sorted list and slice out top 20
    sonnet1_top20 = get_top_k(sonnet1_tf)
    print(f"\nSonnet 1 TF (Top 20):\n{sonnet1_top20}\n")

    # TF of entire corpus
    flattened_corpus = [word for sonnet in corpus.values() for word in sonnet]
    corpus_tf = tf(flattened_corpus)
    corpus_top20 = get_top_k(corpus_tf)
    print(f"Corpus TF (Top 20):\n{corpus_top20}\n")

    # IDF of corpus
    corpus_idf = idf(corpus)
    corpus_tf_ordered = get_top_k(corpus_idf)
    # print top 20 to add to report
    print(f"Corpus IDF (Top 20):\n{corpus_tf_ordered}\n")

    # TFIDF of Sonnet1 w.r.t. corpus
    sonnet1_tfidf = tf_idf(corpus_idf, sonnet1_tf)
    sonnet1_tfidf_ordered = get_top_k(sonnet1_tfidf)
    print(f"Sonnet 1 TFIDF (Top 20):\n{sonnet1_tfidf_ordered}\n")

    # Determine confusion matrix using cosine similarity scores for each exemplar.
    # create matrix of sim scores comparison

