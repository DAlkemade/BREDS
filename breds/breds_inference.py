import fileinput
import logging
import operator
import pickle
from collections import defaultdict
from functools import partial
from typing import List, DefaultDict, Dict

import numpy as np
from box import Box
from nltk.corpus import wordnet as wn
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from visual_size_comparison.config import VisualConfig

from breds.breds import update_tuples_confidences, generate_tuples
from breds.config import Weights, Config
from breds.similarity import similarity_all
from breds.tuple import Tuple
from breds.util import randomString

logger = logging.getLogger(__name__)

N_WORD2VEC = 50
CONTAMINATION_FRAC = .3

def read_weights(parameters_fname: str):
    weights = Weights()
    for line in fileinput.input(parameters_fname):
        if line.startswith("alpha"):
            weights.alpha = float(line.split("=")[1])

        if line.startswith("beta"):
            weights.beta = float(line.split("=")[1])

        if line.startswith("gamma"):
            weights.gamma = float(line.split("=")[1])
    return weights


class BackoffSettings:
    def __init__(self, use_direct = False, use_word2vec: bool = False, use_hypernyms: bool = False, use_hyponyms: bool = False, use_head_noun: bool = False):
        self.use_direct = use_direct
        self.use_word2vec = use_word2vec
        self.use_hypernyms = use_hypernyms
        self.use_hyponyms = use_hyponyms
        self.use_head_noun = use_head_noun

    def print(self) -> str:
        enabled = []
        if self.use_direct:
            enabled.append('direct')
        if self.use_word2vec:
            enabled.append('word2vec')
        if self.use_hyponyms:
            enabled.append('hyponyms')
        if self.use_hypernyms:
            enabled.append('hypernyms')
        if self.use_head_noun:
            enabled.append('head noun')

        logger.info(f'Enabled: {enabled}')
        return str(enabled)




def predict_sizes(all_sizes: dict, objects: list, cfg: BackoffSettings) -> Dict[str, float]:
    """Predict the final size for objects using provided sizes for the objects and their related objects.

    :param all_sizes: dictionary with as keys the objects we are predicting and the values a dict with relevant sizes
    of the object itself and relevant sizes
    """
    predictions = dict()
    for object in objects:
        try:
            sims_dict = all_sizes[object]
        except KeyError:
            logger.warning(f'No sizes for {object}')
            sims_dict = defaultdict(list)


        directs = sims_dict['itself']
        hyponyms = sims_dict['hyponyms']
        hypernyms = sims_dict['hypernyms']
        word2vecs = sims_dict['word2vec']
        head_nouns = sims_dict['head_noun']

        # try:
        #     direct_highest_confidence: Tuple = max(directs, key=lambda item: item.confidence)
        #     logger.info(
        #         f'Direct highest confidence: {direct_highest_confidence.e1} {direct_highest_confidence.e2} with conf {direct_highest_confidence.confidence}')
        # except ValueError:
        #     direct_highest_confidence = None
        size_direct = predict_point(True, [t.e2 for t in directs])

        # TODO instead of taking means, maybe take the mean of the MAX for each hyponym, hypernym, etc
        hyponym_mean = weighted_tuple_mean(hyponyms)

        hypernym_mean = weighted_tuple_mean(hypernyms)

        word2vec_mean = weighted_tuple_mean(word2vecs)

        outlier_detector = EllipticEnvelope(contamination=CONTAMINATION_FRAC)
        sizes_array = np.reshape([t.e2 for t in word2vecs], (-1, 1))
        with np.errstate(all='raise'):
            try:
                # Fit detector
                outlier_detector.fit(sizes_array)
                preds = outlier_detector.predict(sizes_array)
                selected_word2vecs = np.extract(preds == 1, word2vecs)
            except (ValueError, RuntimeWarning, FloatingPointError):
                selected_word2vecs = []

        selected_word2vec_mean = weighted_tuple_mean(selected_word2vecs)


        head_noun_size = predict_point(True, [t.e2 for t in head_nouns])

        # TODO use results in an order, e.g direct finds -> mean of hyponyms -> mean of hypernyms -> word2vec
        #  Maybe this is something I should experiment with

        # TODO maybe at this point do a ttest between the two objects, using their best distribution.
        #  Weigh the importance of each point with their confidences, if possible:
        #  https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttest_ind.html

        # TODO return for each object a size and a confidence
        if size_direct is not None and cfg.use_direct:
            size = size_direct
        elif hyponym_mean is not None and cfg.use_hyponyms:
            size = hyponym_mean
        elif hypernym_mean is not None and cfg.use_hypernyms:
            size = hypernym_mean
        elif head_noun_size is not None and cfg.use_head_noun:
            size = head_noun_size
        elif selected_word2vec_mean is not None and cfg.use_word2vec:
            size = selected_word2vec_mean
        else:
            size = None


        predictions[object] = size

    return predictions


def filter_tuples(candidate_tuples, dev_threshold):
    filtered: DefaultDict[Tuple, list] = defaultdict(list)
    for t, v in candidate_tuples.items():
        if t.confidence > dev_threshold:
            filtered[t].append(v)

    return filtered


def gather_sizes_with_bootstrapping_patterns(cfg: Box, patterns, all_new_objects) -> DefaultDict[Tuple, list]:
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)
    tuples = generate_tuples(randomString(), config, names=all_new_objects)

    config.visual = cfg.parameters.visual_at_inference

    candidate_tuples = extract_tuples(config, patterns, tuples)
    filtered_tuples = filter_tuples(candidate_tuples, cfg.parameters.dev_threshold)

    for t in candidate_tuples.keys():
        logger.info(t.sentence)
        logger.info(f"{t.e1} {t.e2}")
        logger.info(t.confidence)
        logger.info("\n")

    return filtered_tuples


def weighted_tuple_mean(tuples: List[Tuple]):
    try:
        average = np.average([t.e2 for t in tuples], weights=[t.confidence for t in tuples])
    except ZeroDivisionError:
        average = None
    return average


def compile_results(candidate_tuples, objects_lookup, similar_words, unseen_objects):
    # Print tuples
    extracted_tuples = list(candidate_tuples.keys())
    tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence,
                           reverse=True)
    # Compile all results for each object
    all_sizes = defaultdict(partial(defaultdict, list))
    for t in tuples_sorted:
        entity = t.e1
        relevant_objects = objects_lookup[entity]
        for relevant_object in relevant_objects:
            og_dict = similar_words[relevant_object]
            for type, values in og_dict.items():
                if entity in values:
                    all_sizes[relevant_object][type].append(t)

        if entity in unseen_objects:
            all_sizes[entity]['itself'].append(t)
    return all_sizes


def extract_tuples(config, patterns, tuples) -> DefaultDict[Tuple, list]:
    candidate_tuples = defaultdict(list)
    for t in tuples:
        for extraction_pattern in patterns:
            accept, score = similarity_all(
                t, extraction_pattern, config.weights, config.threshold_similarity
            )
            if accept:
                candidate_tuples[t].append(
                    (extraction_pattern, score)
                )
    update_tuples_confidences(candidate_tuples, config)
    return candidate_tuples


def create_reverse_lookup(similar_words):
    objects_lookup = defaultdict(list)
    for main_object, related_words_dict in similar_words.items():
        for type, values in related_words_dict.items():
            for related_word in values:
                objects_lookup[related_word].append(main_object)
    return objects_lookup


def find_similar_words(word2vec_model, unseen_objects):
    similar_words = defaultdict(lambda: defaultdict(list))
    for entity in unseen_objects:
        # Word2vec
        # TODO the results for cheetah and container ship are not great, should probably be a last resort
        # TODO can be sped up if necessary:
        #  https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html#sphx-glr-auto-examples-tutorials-run-annoy-py
        most_similar = word2vec_model.most_similar(positive=entity.split(),
                                                    topn=N_WORD2VEC)  # TODO maybe use a bigram model? Because now those can not be entered and not be given as similar words
        most_similar_filtered = list()
        for sim in most_similar:
            # if sim[1] < .5:
            #     continue
            # check if noun
            word = sim[0]
            synsets = wn.synsets(word.replace(' ', '_'))
            synsets_noun = wn.synsets(word.replace(' ', '_'), pos=wn.NOUN)
            if len(synsets) > 0:
                if len(synsets_noun) > 0:
                    # only append a recognized word if it is a noun
                    most_similar_filtered.append(sim)
            else:
                # append any unknown word of which we don't know the POS
                most_similar_filtered.append(sim)

        most_similar = most_similar_filtered
        # logger.info(most_similar)
        if len(most_similar) > 0:
            words, _ = zip(*most_similar)
        else:
            words = list()
        similar_words[entity]['word2vec'] = words

        # TODO maybe use synset1.path_similarity(synset2)
        # Wordnet children / parents
        synsets = wn.synsets(entity.replace(' ', '_'), pos=wn.NOUN)
        # TODO think about homonyms
        for synset in synsets:
            hypernyms = [s.lemma_names()[0].replace('_', ' ') for s in synset.hypernyms()]  # only use one name
            hyponyms = [s.lemma_names()[0].replace('_', ' ') for s in synset.hyponyms()]
            # TODO set a limit on the number of hyponyms, e.g. 'animal' might have thousands
            # logger.info(synset.lexname())
            similar_words[entity]['hyponyms'] += hyponyms
            similar_words[entity]['hypernyms'] += hypernyms

        words = entity.strip().split(' ')
        head_nouns = list()
        if len(words) > 1:
            head_noun = words[-1]
            head_nouns.append(head_noun)

        similar_words[entity]['head_noun'] = head_nouns
    return similar_words


def predict_using_tuples(tuples_bootstrap, unseen_objects, maximum=True):
    collated = defaultdict(list)
    for t in tuples_bootstrap:

        collated[t.e1].append(t.e2)
        logger.info(f'Confidence: {t.confidence}')

    point_predictions = dict()
    for o in unseen_objects:
        sizes = collated[o]
        size = predict_point(maximum, sizes)
        point_predictions[o] = size
    return point_predictions


def predict_point(maximum, sizes):
    if len(sizes) > 0:
        # TODO think about whether taking the max is good
        if maximum:
            size = max(sizes)
        else:
            size = np.mean(sizes)
    else:
        size = None
    return size


def load_patterns(cfg: Box):
    patterns_paths = cfg.path.patterns
    if cfg.parameters.coreference:
        patterns_paths = patterns_paths.coref
    else:
        patterns_paths = patterns_paths.no_coref
    if cfg.parameters.visual_confidence:
        patterns_fname = patterns_paths.visual
    else:
        patterns_fname = patterns_paths.no_visual
    with open(patterns_fname, 'rb') as f:
        patterns = pickle.load(f)
    return patterns


def load_unseen_objects(cfg):
    unseen_objects_fname = cfg.path.unseen_objects
    unseen_objects = set([line.strip() for line in fileinput.input(unseen_objects_fname)])
    return unseen_objects