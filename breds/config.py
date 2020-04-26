import fileinput
import logging
import re
from typing import Dict

from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from gensim.models import KeyedVectors
from breds.seed import Seed
from breds.reverb import Reverb

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

logger = logging.getLogger(__name__)


class Config(object):

    def __init__(self, config_file, positive_seeds, negative_seeds,
                 similarity, confidence, objects):

        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        # self.filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
        self.filter_pos = []
        # self.regex_clean_simple = re.compile('</?[A-Z]+>', re.U)

        # self.regex_clean_linked = re.compile('</[A-Z]+>|<[A-Z]+ url=[^>]+>', re.U)
        # self.tags_regex = re.compile('</?[A-Z]+>', re.U)
        self.positive_seed_tuples: Dict[str, Seed] = dict()
        self.negative_seed_tuples: Dict[str, Seed] = dict()
        self.vec_dim = 0
        self.e1_type = None
        self.e2_type = None
        self.stopwords = stopwords.words('english')
        self.lmtzr = WordNetLemmatizer()
        self.threshold_similarity = similarity
        self.instance_confidence = confidence
        self.reverb = Reverb()
        self.word2vec = None
        self.vec_dim = None

        # simple tags, e.g.:
        # <PER>Bill Gates</PER>
        self.regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

        # linked tags e.g.:
        # <PER url=http://en.wikipedia.org/wiki/Mark_Zuckerberg>Zuckerberg</PER>
        self.regex_linked = re.compile('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', re.U)

        self.objects = read_objects_of_interest(objects)
        logger.info(f'Number of objects: {len(self.objects)}')

        #TODO clean up config file stuff and use the config library
        for line in fileinput.input(config_file):
            if line.startswith("#") or len(line) == 1:
                continue

            if line.startswith("wUpdt"):
                self.wUpdt = float(line.split("=")[1])

            if line.startswith("wUnk"):
                self.wUnk = float(line.split("=")[1])

            if line.startswith("wNeg"):
                self.wNeg = float(line.split("=")[1])

            if line.startswith("number_iterations"):
                self.number_iterations = int(line.split("=")[1])

            if line.startswith("min_pattern_support"):
                self.min_pattern_support = int(line.split("=")[1])

            if line.startswith("max_tokens_away"):
                self.max_tokens_away = int(line.split("=")[1])

            if line.startswith("min_tokens_away"):
                self.min_tokens_away = int(line.split("=")[1])

            if line.startswith("context_window_size"):
                self.context_window_size = int(line.split("=")[1])

            if line.startswith("similarity"):
                self.similarity = line.split("=")[1].strip()

            if line.startswith("word2vec_path"):
                self.word2vecmodelpath = line.split("=")[1].strip()

            if line.startswith("alpha"):
                self.alpha = float(line.split("=")[1])

            if line.startswith("beta"):
                self.beta = float(line.split("=")[1])

            if line.startswith("gamma"):
                self.gamma = float(line.split("=")[1])

            if line.startswith("tags_type"):
                self.tag_type = line.split("=")[1].strip()
                if self.tag_type != 'simple':
                    raise RuntimeWarning('tags_type not supported')

            if line.startswith("relative_difference_cutoff"):
                self.relative_difference_cutoff = float(line.split("=")[1].strip())

            if line.startswith("coreference"):
                cor_string = line.split("=")[1].strip()
                self.coreference = cor_string == 'True'


        assert self.alpha+self.beta+self.gamma == 1

        self.read_seeds(positive_seeds, self.positive_seed_tuples)
        self.read_seeds(negative_seeds, self.negative_seed_tuples)
        fileinput.close()

        logger.info("Configuration parameters")
        logger.info("========================\n")

        logger.info("Relationship/Sentence Representation")
        logger.info(f"e1 type              : {self.e1_type}")
        logger.info(f"e2 type              : { self.e2_type}")
        logger.info(f"tags type             { self.tag_type}")
        logger.info(f"context window        { self.context_window_size}")
        logger.info(f"max tokens away       { self.max_tokens_away}")
        logger.info(f"min tokens away       { self.min_tokens_away}")
        logger.info(f"Word2Vec Model        { self.word2vecmodelpath}")

        logger.info("\nContext Weighting")
        logger.info(f"alpha                 { self.alpha}")
        logger.info(f"beta                  { self.beta}")
        logger.info(f"gamma                 { self.gamma}")

        logger.info("\nSeeds")
        logger.info(f"positive seeds        { len(self.positive_seed_tuples)}")
        logger.info(f"negative seeds        { len(self.negative_seed_tuples)}")
        logger.info(f"negative seeds wNeg   { self.wNeg}")
        logger.info(f"unknown seeds wUnk    { self.wUnk}")

        logger.info("\nParameters and Thresholds")
        logger.info(f"threshold_similarity  { self.threshold_similarity}")
        logger.info(f"instance confidence   { self.instance_confidence}")
        logger.info(f"min_pattern_support   { self.min_pattern_support}")
        logger.info(f"iterations            { self.number_iterations}")
        logger.info(f"iteration wUpdt       { self.wUpdt}")
        logger.info(f"Coreference:          {self.coreference}")
        logger.info("\n")

    def read_word2vec(self):
        logger.info("Loading word2vec model ...\n")
        self.word2vec = KeyedVectors.load_word2vec_format(self.word2vecmodelpath, binary=True)
        self.vec_dim = self.word2vec.vector_size
        logger.info(self.vec_dim, "dimensions")

    def read_seeds(self, seeds_file, holder: Dict[str, Seed]):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e1 = e1.lower()
                e2 = line.split(";")[1].strip()
                e2 = float(e2)
                self.objects.add(e1)
                self.add_seed_to_dict(e1, e2, holder)

    def add_seed_to_dict(self, e1: str, size: float, seed_dict: Dict[str, Seed]):
        """Add seed to seeds.

        :param e1:
        :param size:
        :param seed_dict:
        :return: True if a new seed was added
        """
        try:
            seed_dict[e1].add_size(size)
            return False
        except KeyError:
            seed_dict[e1] = Seed(e1, size)
            return True


def read_objects_of_interest(objects_path):
    objects = set()
    for line in fileinput.input(objects_path):
        object = line.strip().lower()
        objects.add(object)
    return objects