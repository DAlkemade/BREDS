import fileinput
import logging
import re
from typing import Dict

from box import Box
from gensim.models import KeyedVectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from breds.reverb import Reverb
from breds.seed import Seed
from visual_size_comparison.config import VisualConfig

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

logger = logging.getLogger(__name__)

class Weights():
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None

class Config(object):

    def __init__(self, cfg: Box, visual_config: VisualConfig):

        # http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        # select everything except stopwords, ADJ and ADV
        # self.filter_pos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']
        self.visual_config = visual_config
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
        self.threshold_similarity = cfg.parameters.similarity
        self.instance_confidence = cfg.parameters.confidence
        self.reverb = Reverb()
        self.word2vec = None
        self.vec_dim = None
        self.weights = Weights()

        # simple tags, e.g.:
        # <PER>Bill Gates</PER>
        self.regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)

        # linked tags e.g.:
        # <PER url=http://en.wikipedia.org/wiki/Mark_Zuckerberg>Zuckerberg</PER>
        self.regex_linked = re.compile('<[A-Z]+ url=[^>]+>[^<]+</[A-Z]+>', re.U)

        self.objects = read_objects_of_interest(cfg.path.objects)
        logger.info(f'Number of objects: {len(self.objects)}')


        self.wUpdt = float(cfg.parameters.wUpdt)
        self.wUnk = float(cfg.parameters.wUnk)
        self.wNeg = float(cfg.parameters.wNeg)
        self.number_iterations = int(cfg.parameters.number_iterations)
        self.min_pattern_support = int(cfg.parameters.min_pattern_support)
        self.max_tokens_away = int(cfg.parameters.max_tokens_away)
        self.min_tokens_away = int(cfg.parameters.min_tokens_away)
        self.context_window_size = int(cfg.parameters.context_window_size)
        self.word2vecmodelpath = cfg.parameters.word2vec_path
        self.weights.alpha = float(cfg.parameters.alpha)
        self.weights.beta = float(cfg.parameters.beta)
        self.weights.gamma = float(cfg.parameters.gamma)
        self.htmls_cache = str(cfg.path.htmls_cache)
        self.htmls_cache_coref = str(cfg.path.htmls_cache_coref)

        self.tag_type = cfg.parameters.tags_type
        if self.tag_type != 'simple':
            raise RuntimeWarning('tags_type not supported')

        self.relative_difference_cutoff = float(cfg.parameters.relative_difference_cutoff)


        self.coreference: bool = cfg.parameters.coreference
        assert type(self.coreference) is bool


        self.visual: bool = cfg.parameters.visual_confidence
        assert type(self.visual) is bool

        self.visual_cutoff: float = float(cfg.parameters.visual_cutoff)


        assert self.weights.alpha+self.weights.beta+self.weights.gamma == 1

        self.read_seeds(cfg.path.seeds_file, self.positive_seed_tuples)
        self.read_seeds(cfg.path.negative_seeds, self.negative_seed_tuples)
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
        logger.info(f"alpha                 { self.weights.alpha}")
        logger.info(f"beta                  { self.weights.beta}")
        logger.info(f"gamma                 { self.weights.gamma}")

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
        logger.info(f"Visual:               {self.visual}")
        logger.info(f"Visual cutoff:        {self.visual_cutoff}")

        logger.info("\n")

    def read_word2vec(self):
        logger.info("Loading word2vec model ...\n")
        self.word2vec = load_word2vec(self.word2vecmodelpath)
        self.vec_dim = self.word2vec.vector_size
        logger.info(f"{self.vec_dim} dimensions")

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


def load_word2vec(word2vecmodelpath: str):
    return KeyedVectors.load_word2vec_format(word2vecmodelpath, binary=True)


def read_objects_of_interest(objects_path) -> set:
    objects = set()
    for line in fileinput.input(objects_path):
        object = line.strip().lower()
        objects.add(object)
    return objects


def parse_objects_from_seed(seeds_file) -> set:
    objects = set()
    for line in fileinput.input(seeds_file):
        if line.startswith("#") or len(line) == 1:
            continue
        if line.startswith("e1"):
            continue
        elif line.startswith("e2"):
            continue
        else:
            e1 = line.split(";")[0].strip().lower()
            objects.add(e1)
    return objects