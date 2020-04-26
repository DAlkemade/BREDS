import logging
import uuid
import numpy as np
from size_comparisons.inference.baseline_numeric_gaussians import BaselineNumericGaussians

from breds.config import Config

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

logger = logging.getLogger(__name__)

class Pattern(object):

    def __init__(self, t=None):
        self.id = uuid.uuid4()
        self.positive = 0
        self.negative = 0
        self.unknown = 0
        self.confidence = 0
        self.tuples = set()
        self.bet_uniques_vectors = set()
        self.bet_uniques_words = set()
        if t is not None:
            self.tuples.add(t)

    def __eq__(self, other):
        return self.tuples == other.tuples

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def update_confidence(self, config):
        if self.positive > 0:
            self.confidence = (
                float(self.positive) / float(self.positive +
                                             self.unknown * config.wUnk +
                                             self.negative * config.wNeg
                                             )
            )
        elif self.positive == 0:
            self.confidence = 0

    def add_tuple(self, t):
        self.tuples.add(t)

    def merge_all_tuples_bet(self):
        """
        Put all tuples with BET vectors into a set so that comparison with repeated vectors
        is eliminated
        """
        self.bet_uniques_vectors = set()
        self.bet_uniques_words = set()
        for t in self.tuples:
            # transform numpy array into a tuple so it can be hashed and added into a set
            self.bet_uniques_vectors.add(tuple(t.bet_vector))
            self.bet_uniques_words.add(t.bet_words)

    def update_selectivity(self, t, config: Config):
        matched_both = False
        matched_e1 = False

        #TODO think about whether it's fair that EVERY value for a seed counts equally. maybe it should use a mean from all the sizes in seed.sizes?
        for s in config.positive_seed_tuples.values():
            if s.e1.strip() == t.e1.strip():
                matched_e1 = True
                tuple_number = t.e2
                for seed_size in s.sizes:

                    # TODO this should be less crude. use the difference in the confidence value
                    if abs((tuple_number - seed_size) / seed_size) < config.relative_difference_cutoff:
                        self.positive += 1
                        matched_both = True
                        break

        if matched_e1 is True and matched_both is False:
            self.negative += 1

        if matched_both is False:
            for n in config.negative_seed_tuples.values():
                if n.e1.strip() == t.e1.strip():
                    for seed_size in n.sizes:
                        if seed_size == t.e2.strip():
                            self.negative += 1
                            matched_both = True
                            break

        if not matched_both and not matched_e1:
            self.unknown += 1
