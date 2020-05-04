import logging
import uuid
from typing import List

from breds.config import Config
from breds.visual import check_tuple_with_visuals

import numpy as np

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

logger = logging.getLogger(__name__)


def pattern_factory(config, t=None):
    if config.visual:
        return PatternVisual(t)
    else:
        return Pattern(t)


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

        tuple_e1 = t.e1.strip()
        tuple_number = t.e2

        #TODO think about whether it's fair that EVERY value for a seed counts equally. maybe it should use a mean from all the sizes in seed.sizes?
        for s in config.positive_seed_tuples.values():
            if s.e1.strip() == tuple_e1:
                matched_e1 = True
                for seed_size in s.sizes:

                    # TODO this should be less crude. use the difference in the confidence value
                    # positive if ANY of the values in the list match
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


class PatternVisual(Pattern):
    def __init__(self, t=None):
        super().__init__(t)
        self.visual_results: List[bool] = []

    def update_selectivity(self, t, config: Config):
        super().update_selectivity(t, config)

        # TODO problem: I use the max bounding box sizes, but some valid patterns return another dimension, i.e. 'height' pattern will be penalized for 'jaguar', because the jaguar is much wider than it is high on a picture
        tuple_e1 = t.e1.strip()
        tuple_number = t.e2

        self.visual_results += check_tuple_with_visuals(config.visual_config, tuple_e1, tuple_number)

        # TODO might be a bit distorted, because getting a negative is harder, because for all synsets enough comparisons need to be available


    def update_confidence(self, config):
        super().update_confidence(config)
        total_hits = len(self.visual_results)
        if total_hits > 100: # only use visual if enough comparisons were used
            visual_confidence = np.mean(self.visual_results)
            logger.info(f'Used visual: {total_hits} hits; visual confidence {visual_confidence}; normal confidence: {self.confidence}')
            if visual_confidence < config.visual_cutoff: # TODO think about visual cutoff
                self.confidence = 0.
        else:
            logger.info(f'Did NOT use visual for the confidence of pattern')


