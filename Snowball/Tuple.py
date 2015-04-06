# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
from nltk import PunktWordTokenizer, pos_tag
from reverb.ReVerb import Reverb


class Tuple(object):

        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']

        def __init__(self, _e1, _e2, _sentence, _before, _between, _after, config):
            self.e1 = _e1
            self.e2 = _e2
            self.sentence = _sentence
            self.confidence = 0
            self.confidence_old = 0
            self.bef_words = _before
            self.bet_words = _between
            self.aft_words = _after
            self.config = config
            self.bef_vector = None
            self.bet_vector = None
            self.aft_vector = None

            self.bef_reverb_vector = None
            self.bet_reverb_vector = None
            self.aft_reverb_vector = None
            self.passive_voice = None

            if config.reverb is True:
                # construct TF-IDF vectors with the words part of a ReVerb pattern
                # or if no ReVerb patterns with selected words from the contexts
                self.extract_patterns(config)

            elif config.reverb is False:
                self.bef_vector = self.create_vector(self.bef_words)
                self.bet_vector = self.create_vector(self.bet_words)
                self.aft_vector = self.create_vector(self.aft_words)

        def get_vector(self, context):
            if context == "bef":
                return self.bef_vector
            elif context == "bet":
                return self.bet_vector
            elif context == "aft":
                return self.aft_vector
            else:
                print "Error, vector must be 'bef', 'bet' or 'aft'"
                sys.exit(0)

        def create_vector(self, text):
            vect_ids = self.config.vsm.dictionary.doc2bow(self.tokenize(text))
            return self.config.vsm.tf_idf_model[vect_ids]

        def tokenize(self, text):
            return [word for word in PunktWordTokenizer().tokenize(text.lower()) if word not in self.config.stopwords]

        def __str__(self):
            return str(self.bef_words.encode("utf8")+' '+self.bet_words.encode("utf8")+' '+self.aft_words.encode("utf8"))

        def __eq__(self, other):
            return (self.e1 == other.e1 and self.e2 == other.e2 and self.bef_words == other.bef_words and
                    self.bet_words == other.bet_words and self.aft_words == other.aft_words)

        @staticmethod
        def detect_passive_voice(config, pattern):
            aux_verbs = ['be']
            for i in range(0, len(pattern)):
                # TODO: contar com adjectivos pelo meio
                if pattern[i][1].startswith('V'):
                    verb = config.lmtzr.lemmatize(pattern[i][0], 'v')
                    if verb in aux_verbs and i + 2 <= len(pattern) - 1:
                        if (pattern[i + 1][1] == 'VBN' or pattern[i + 1][1] == 'VBD') and pattern[i + 2][0] == 'by':
                            return True
                        else:
                            return False

        def construct_pattern_vector(self, pattern_tags, config):
            # construct TF-IDF representation for each context
            pattern = [t[0] for t in pattern_tags[0] if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
                return self.config.vsm.tf_idf_model[vect_ids]

        def construct_words_vectors(self, words, config):
            # split text into tokens and tag them using NLTK's default English tagger
            # POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            text_tokens = PunktWordTokenizer().tokenize(words)
            tags_ptb = pos_tag(text_tokens)
            pattern = [t[0] for t in tags_ptb if t[0].lower() not in config.stopwords and t[1] not in self.filter_pos]
            if len(pattern) >= 1:
                vect_ids = self.config.vsm.dictionary.doc2bow(pattern)
                return self.config.vsm.tf_idf_model[vect_ids]

        def extract_patterns(self, config):
            patterns_bef, patterns_bef_tags = Reverb.extract_reverb_patterns_ptb(self.bef_words)
            patterns_bet, patterns_bet_tags = Reverb.extract_reverb_patterns_ptb(self.bet_words)
            patterns_aft, patterns_aft_tags = Reverb.extract_reverb_patterns_ptb(self.aft_words)

            # detect passive voice in BET ReVerb pattern
            if len(patterns_bet_tags) > 0:
                self.passive_voice = self.detect_passive_voice(config, patterns_bet_tags[0])

            # BEF context
            if len(patterns_bef_tags) > 0:
                self.bef_vector = self.construct_pattern_vector(patterns_bef_tags, config)
            else:
                self.bef_vector = self.construct_words_vectors(self.bef_words, config)

            # BET context
            if len(patterns_bet_tags) > 0:
                self.bet_vector = self.construct_pattern_vector(patterns_bet_tags, config)
            else:
                self.bet_vector = self.construct_words_vectors(self.bet_words, config)

            # AFT context
            if len(patterns_aft_tags) > 0:
                self.aft_vector = self.construct_pattern_vector(patterns_aft_tags, config)
            else:
                self.aft_vector = self.construct_words_vectors(self.aft_words, config)

            """
            print self.e1
            print self.e2
            print self.sentence
            print "ReVerb Patterns BEF:", len(patterns_bef_tags)
            print "ReVerb Patterns BET:", len(patterns_bet_tags)
            print "ReVerb Patterns AFT:", len(patterns_aft_tags)
            if len(patterns_bef_tags) > 0:
                print patterns_bef_tags[0]
            if len(patterns_bet_tags) > 0:
                print patterns_bet_tags[0]
            if len(patterns_aft_tags) > 0:
                print patterns_aft_tags[0]
            print "BEF:", self.bef_vector
            print "BET:", self.bet_vector
            print "AFT:", self.aft_vector
            """