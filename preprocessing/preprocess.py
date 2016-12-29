#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 28, 2016
.. codeauthor: svitlana vakulenko
<svitlana.vakulenko@gmail.com>

Build corpus of tweets
'''

from twython import Twython
from gensim import corpora

# load libraries for text pre-processing
from collections import defaultdict, Counter
import nltk
import langid
import re
from twokenize import tokenizeRawTweetText, punctSeq
from nltk.stem import WordNetLemmatizer

# load Twitter API access keys
from twitter_settings import *
# load stopword lists
from stoplist_twitter import STOPLIST_TW
from frequent_words import STOPLIST

import unittest


def get_hashtag_tweets(hashtag, ndocs, result_type, language_code, corpus_filename, dict_filename):
    '''
    Filter out only the original tweets excluding replies and retweets
    type: recent, popular, mixed
    ndocs (max 100)
    '''
    # connect to Twitter Search API
    twitter_client = Twython(APP_KEY, APP_SECRET,
                             OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

    request = hashtag + " -filter:retweets AND -filter:replies"
    # retrieve tweets
    tweets = twitter_client.search(q=request, result_type=result_type, include_entities='true',
                                   count=ndocs, lang=language_code)['statuses']
    documents = [tw['text'] for tw in tweets]
    print(str(len(documents)) + " input documents")
    generate_corpus(documents, dict_filename, corpus_filename, documents)


def preprocess(documents):
    # Remove urls: http? bug fixed (SV)
    documents = [re.sub(r"(?:\@|https?\://)\S+", "", doc)
                 for doc in documents]

    # Remove documents with less 100 words (some tweets contain only URLs)
    # documents = [doc for doc in documents if len(doc) > 100]

    # Tokenize
    documents = [tokenizeRawTweetText(doc.lower()) for doc in documents]
    # print documents

    # Remove stop words
    unigrams = [w for doc in documents for w in doc if len(w) == 1]
    bigrams = [w for doc in documents for w in doc if len(w) == 2]
    # print bigrams
    stoplist = set(nltk.corpus.stopwords.words(
                   "english") + STOPLIST_TW + STOPLIST + unigrams + bigrams)
    # and strip #
    documents = [[token.lstrip('#') for token in doc if token not in stoplist]
                 for doc in documents]

    # remove punctuation tokens
    documents = [[token for token in doc if not re.match(punctSeq, token)]
                 for doc in documents]

    # Remove words that only occur once
    token_frequency = defaultdict(int)

    lmtzr = WordNetLemmatizer()
    documents = [[lmtzr.lemmatize(token) for token in doc]
                 for doc in documents]

    # count all token
    for doc in documents:
        for token in doc:
            token_frequency[token] += 1

    # keep words that occur more than once
    documents = [[token for token in doc if token_frequency[token] > 1]
                 for doc in documents]

    # print documents
    return documents


def generate_corpus(documents, dict_filename, corpus_filename, tweets, show_documents=True):
    documents = preprocess(documents)
    # Sort words in documents
    for doc in documents:
        doc.sort()
    # format corpus and dictionary for Gensim:
    # Build a dictionary where for each document each word has its own id
    dictionary = corpora.Dictionary(documents)
    dictionary.compactify()

    # Build the corpus: vectors with occurence of each word for each document
    # convert tokenized documents to vectors
    # filter out empty bags and do not add them to the corpus
    corpus = [dictionary.doc2bow(doc) for doc in documents if doc]

    ##########################################
    print(str(len(corpus)) +
          " documents left after pre-processing ")
    print(dictionary)
    # print corpus
    # save the dictionary for future use
    dictionary.save(dict_filename)
    # show only tweets that do not have an empty bag and were used for modeling
    if show_documents:
        counter = 0
        for ind in xrange(len(documents)):
            if documents[ind]:
                print counter, tweets[ind].replace('\n', ' ')
                counter += 1
    # print documents
    # save in Market Matrix format
    corpora.MmCorpus.serialize(corpus_filename, corpus)


def get_user_tweets(user, ndocs,
                    corpus_filename, dict_filename):
    # connect to Twitter Search API
    twitter_client = Twython(APP_KEY, APP_SECRET,
                             OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

    # retrieve user timeline (our corpus)
    user_timeline = twitter_client.get_user_timeline(screen_name=user,
                                                     count=ndocs)

    # tweet preprocessing:
    # Filter out non-english tweets
    documents = [tw['text'] for tw in user_timeline
                 if ('lang' in tw.keys()) and (tw['lang'] in ('en', 'und'))]
    print(str(len(documents)) + " input documents")
    generate_corpus(documents, dict_filename, corpus_filename, documents)

class TestCorpusGenerators(unittest.TestCase):
    def test_get_user_tweets(self):
        user = 'AxelPolleres'
        ndocs = 100
        data_path = '../data/'
        corpus_filename = data_path + user + str(ndocs) + '.mm'
        dict_filename = data_path + user + str(ndocs) + '.dict'
        get_user_tweets('@' + user, ndocs,
                        corpus_filename, dict_filename)
        corpus = corpora.MmCorpus(corpus_filename)
        dictionary = corpora.Dictionary.load(dict_filename)

    def test_get_hashtag_tweets(self):
        hashtag = 'PieceofthePieContest'
        ndocs = 100
        data_path = '../data/'
        corpus_filename = data_path + hashtag + str(ndocs) + '.mm'
        dict_filename = data_path + hashtag + str(ndocs) + '.dict'
        get_hashtag_tweets('#'+hashtag, ndocs, 'mixed', 'en',
                           corpus_filename, dict_filename)
        corpus = corpora.MmCorpus(corpus_filename)
        dictionary = corpora.Dictionary.load(dict_filename)


if __name__ == '__main__':
    unittest.main()
