#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 29, 2016
.. codeauthor: svitlana vakulenko
<svitlana.vakulenko@gmail.com>

Fetching and clustering a bunch of tweets that contain a specified hashtag
'''

import os

from gensim import corpora

from preprocessing.preprocess import get_hashtag_tweets
from cluster import cluster_corpus


def load_corpus(hashtag, ndocs):
    data_path = 'data/'
    corpus_filename = data_path + hashtag + str(ndocs) + '.mm'
    dict_filename = data_path + hashtag + str(ndocs) + '.dict'
    try:
        corpus = corpora.MmCorpus(corpus_filename)
        # print len(corpus)
        dictionary = corpora.Dictionary.load(dict_filename)
    except:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        corpus, dictionary = get_hashtag_tweets('#'+hashtag, ndocs, 'mixed', 'en',
                                                corpus_filename, dict_filename)
    return (corpus, dictionary)


def test_check_hashtag():
    '''
    Maximum 100 docs per a single call to the Twitter API
    '''
    hashtag = 'brexit'
    ndocs = 100
    corpus, dictionary = load_corpus(hashtag, ndocs)
    clusters = cluster_corpus(corpus, dictionary)


if __name__ == '__main__':
    test_check_hashtag()