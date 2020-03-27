import uuid
import numpy as np

from size_comparisons.inference.baseline_numeric_gaussians import load_and_update_baseline, BaselineNumericGaussians

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Pattern(object):

    def __init__(self, t=None):
        self.id = uuid.uuid4()
        self.p_values = list() # TODO should this be wiped every iteration? Or will a new object be generated
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
        if self.p_values is None:
            raise ValueError('First compute confidences for each tuple')

        mean_p_value = np.mean(self.p_values)
        self.confidence = 1 - mean_p_value


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

    def update_selectivity(self, t, config, numeric_seed: BaselineNumericGaussians):
        tuple_pvalue = numeric_seed.shortest_path(t.e1, t.e2)
        self.p_values.append(tuple_pvalue)
