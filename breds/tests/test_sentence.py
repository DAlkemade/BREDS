from unittest import TestCase
from unittest.mock import Mock

from nltk.data import load

from breds.sentence import Sentence


class TestPattern(TestCase):

    def setUp(self):
        self.config = Mock()
        attrs = {"tag_type": 'simple', "other.side_effect": KeyError}
        self.config.configure_mock(**attrs)
        print(self.config.tag_type)
        self.tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')

    def test_update_selectivity(self):
        sentence = Sentence('A tiger is 2.1 meters', 'OBJECT', 'NUMBER', 20, 0, 2, 'tiger', self.tagger, self.config)
        self.assertEqual(len(sentence.relationships), 1)
        print(sentence.relationships)
