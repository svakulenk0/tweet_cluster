#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 28, 2016
.. codeauthor: svitlana vakulenko
<svitlana.vakulenko@gmail.com>

Clustering tweets
'''

from preprocessing.preprocess import get_corpus

from collections import Counter
import numpy
import os

from gensim import corpora, models, matutils
import fastcluster
from scipy.cluster.hierarchy import fcluster


def generate_tfidf(corpus):
    tfidf = models.tfidfmodel.TfidfModel(corpus)
    return tfidf[corpus]


def hclust(X, max_d=1.0, distance_metric='euclidean'):
    # print X
    print X.shape
    # cluster_ids = fclusterdata(X, max_d, criterion='distance')
    HL = fastcluster.linkage(X, method='average', metric=distance_metric)
    # print HL
    print "max_d = ", max_d
    cluster_ids = fcluster(HL, max_d, criterion='distance')
    freqTwCl = Counter(cluster_ids)
    print "n_clusters:", len(freqTwCl)
    npindL = numpy.array(cluster_ids)
    clusters = []
    for cl, freq in freqTwCl.most_common():
        clidx = (npindL == cl).nonzero()[0].tolist()
        clusters.append(clidx)
    return clusters


def test_hclust():
    '''
    Demonstrates how to cluster tweets retrieved from the timeline of a specified Twitter user,
    e.g. @realDonaldTrump @HillaryClinton @AxelPolleres @webLyzard
    '''
    user = 'realDonaldTrump'
    ndocs = 20
    data_path = 'data/'
    corpus_filename = data_path + user + str(ndocs) + '.mm'
    dict_filename = data_path + user + str(ndocs) + '.dict'
    try:
        corpus = corpora.MmCorpus(corpus_filename)
        # print len(corpus)
        dictionary = corpora.Dictionary.load(dict_filename)
    except:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        get_corpus('@' + user, ndocs,
                   corpus_filename, dict_filename, show_documents=True, filter_eng=False)
    corpus_tfidf = generate_tfidf(corpus)
    array = matutils.corpus2csc(corpus_tfidf).toarray()
    X = array.T
    clusters = hclust(X)
    print "Clusters:", clusters
    for cluster in clusters:
        print Counter([dictionary[word] for doc_id in cluster for word, score in corpus_tfidf[doc_id]])


if __name__ == '__main__':
    test_hclust()