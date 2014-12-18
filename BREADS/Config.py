#!/usr/bin/env python
# -*- coding: utf-8 -*-
from BREADS import Seed

__author__ = 'dsbatista'

import fileinput
from gensim.models import Word2Vec


class Config(object):

    def __init__(self, config_file, seeds_file):

        self.seed_tuples = set()
        self.vec_dim = 0

        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("use_RlogF"):
                self.use_RlogF = bool(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("threshold_similarity"):
                self.threshold_similarity = float(line.split("=")[1])

            if line.startswith("instance_confidance"):
                self.instance_confidance = float(line.split("=")[1])

            if line.startswith("single_vector"):
                self.single_vector = line.split("=")[1]

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1]

            if line.startswith("word2vec_path"):
                self.word2vecmodelpath = line.split("=")[1].strip()

        print "Loading word2vec model ...\n"
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = 200
        self.read_seeds(self, seeds_file)
        fileinput.close()

    @staticmethod
    def read_seeds(self, seeds_file):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                self.seed_tuples.add(seed)
        print len(self.seed_tuples), "seeds instances loaded"