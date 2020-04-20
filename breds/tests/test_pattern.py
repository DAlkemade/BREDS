from collections import namedtuple
from unittest import TestCase
from unittest.mock import MagicMock
import pandas as pd

from scipy.stats import norm
from size_comparisons.inference.baseline_numeric_gaussians import load_and_update_baseline, BaselineNumericGaussians

from breds.pattern import Pattern
from breds.seed import Seed
from breds.tuple import Tuple


class TestPattern(TestCase):

    def setUp(self):
        raise NotImplementedError('Tests have not been updated for numeric')
        seed_set = set()
        Entry = namedtuple('Entry', ['name', 'sizes'])
        data_list = list()
        self.e1 = 'seed_1'
        self.e2 = 'seed_2'
        self.e3 = 'seed_3'
        self.e4 = 'seed_4'
        data_list.append(Entry(self.e1, norm.rvs(1000.0, 2.5, size=500)))
        data_list.append(Entry(self.e2, norm.rvs(100., .01, size=100)))
        data_list.append(Entry(self.e3, norm.rvs(40., .01, size=2)))
        data_list.append(Entry(self.e4, norm.rvs(50., 50., size=5)))
        data = pd.DataFrame(data_list)
        self.baseline = BaselineNumericGaussians(data)
        self.baseline.fill_adjacency_matrix()
        self.baseline.update_distance_matrix()

        # TODO maybe start using seed graph (or the most confident edges), but might not be smart to start with something noisy
        seed_set.add(Seed(self.e1, self.e2))
        # seed_set.add(Seed(self.e3, self.e4))

        self.config = MagicMock()
        self.config.positive_seed_tuples = seed_set

    def test_update_selectivity(self):

        bef_words = ['dummy']
        bet_words = ['dummy']
        aft_words = ['dummy']

        # positive
        pattern = Pattern()
        t = Tuple(self.e1, self.e2, None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config, self.baseline)
        self.assertEqual(len(pattern.p_values), 1)
        self.assertEqual(pattern.p_values[0], self.baseline.shortest_path(self.e1, self.e2))


    def test_update_confidence(self):
        bef_words = ['dummy']
        bet_words = ['dummy']
        aft_words = ['dummy']

        # positive
        pattern = Pattern()
        t = Tuple(self.e1, self.e2, None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config, self.baseline)
        pattern.update_confidence(self.config)
        print(pattern.p_values[0])
        self.assertGreater(pattern.confidence, .5)

        # negative
        pattern = Pattern()
        t = Tuple(self.e2, self.e1, None, bef_words, bet_words, aft_words, self.config)
        pattern.update_selectivity(t, self.config, self.baseline)
        pattern.update_confidence(self.config)
        self.assertLess(pattern.confidence, .5)
