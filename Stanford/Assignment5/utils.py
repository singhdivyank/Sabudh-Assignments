#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2019-20: Homework 5
utils.py:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')


def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and longest words in all sentences.
    @param sents: list of sentences, result of `words2charindices() `from `vocab.py` -- list[list[list[int]]]
    @param char_pad_token: index of the character-padding token -- int
    @returns sents_padded: list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters -- list[list[list[int]]]
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s)
    max_sent_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for k in range(batch_size):
        sentence = sents[k]
        sent_padded = []

        for w in sentence:
            data = [c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))]
            if len(data) > max_word_length:
                data = data[:max_word_length]
            sent_padded.append(data)

        sent_padded = sent_padded[:max_sent_len] + [[char_pad_token]*max_word_length] * max(0,
                                                                                            max_sent_len -
                                                                                            len(sent_padded))
        sents_padded.append(sent_padded)

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents: list of sentences, where each sentence
                                    is represented as a list of words -- list[list[int]]
    @param pad_token: padding token -- int
    @returns sents_padded: list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length -- list[list[int]]
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    # COPY OVER YOUR CODE FROM ASSIGNMENT 4

    length = max(len(s) for s in sents)
    for s in sents:
        sentence_length = len(s)
        sents_padded.append(s + (length - sentence_length) * [pad_token])

    # END YOUR CODE FROM ASSIGNMENT 4

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path: path to file containing corpus -- str
    @param source: "tgt" or "src" indicating whether text is
                    of the source language or target language -- str
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data: list of tuples containing source and target sentence -- list of (src_sent, tgt_sent)
    @param batch_size: batch size -- int
    @param shuffle: whether to randomly shuffle the dataset -- boolean
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

