__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

from numpy import zeros
from Word2VecWrapper import Word2VecWrapper
from math import log


class Pattern(object):

    def __init__(self, t=None):
        self.single_vector = zeros(200)
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.tuples = set()
        self.patterns_words = set()
        if tuple is not None:
            self.tuples.add(t)
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def __hash__(self):
        return hash((self.patterns_words, self.tuples))

    def __eq__(self, other):
        return (self.tuples, self.patterns_words) == (other.tuples, other.patterns_words)

    def __str__(self):
        return " | ".join([p for p in self.patterns_words]).encode("utf8")

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence_2003(self, config):
        if self.positive > 0:
            self.confidence = log(float(self.positive), 2) * (float(self.positive) / float(self.positive + self.unknown * config.wUnk + self.negative * config.wNeg))
        elif self.positive == 0:
            self.confidence = 0

    def update_confidence(self):
        if self.positive > 0 or self.negative > 0:
            self.confidence = float(self.positive) / float(self.positive + self.negative)

    def add_tuple(self, t):
        self.tuples.add(t)

    def merge_patterns(self):
        for t in self.tuples:
            for p in t.patterns_words:
                self.patterns_words.add(p)

    def calculate_single_vector(self):
        self.merge_patterns()
        pattern_vector = zeros(200)
        for p in self.patterns_words:
            vector_p = Word2VecWrapper.pattern2vector(p, 200)
            pattern_vector += vector_p
        self.single_vector = pattern_vector

    def update_selectivity(self, t, config):
        for s in config.seed_tuples:
            if s.e1 == t.e1 or s.e1.strip() == t.e1.strip():
                if s.e2 == t.e2.strip() or s.e2.strip() == t.e2.strip():
                    self.positive += 1
                else:
                    self.negative += 1
            else:
                for n in config.negative_seed_tuples:
                    if n.e1 == t.e1 or n.e1.strip() == t.e1.strip():
                        if n.e2 == t.e2.strip() or n.e2.strip() == t.e2.strip():
                            self.negative += 1
                self.unknown += 1

        #self.update_confidence()
        self.update_confidence_2003(config)
