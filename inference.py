import fileinput
import logging
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
from datetime import datetime

from gensim.models import Word2Vec
from logging_setup_dla.logging import set_up_root_logger

from breds.breds import process_objects, update_tuples_confidences
from breds.config import Weights, Config
from breds.htmls import scrape_htmls
from breds.similarity import similarity_all
from breds.visual import VisualConfig

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.getcwd())

logger = logging.getLogger(__name__)

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

def main():
    parser = ArgumentParser()
    parser.add_argument('--patterns', type=str, required=True)
    parser.add_argument('--unseen_objects', type=str, required=True)
    parser.add_argument('--configuration', type=str, required=True)
    parser.add_argument('--seeds_file', type=str, required=True)
    parser.add_argument('--negative_seeds', type=str, required=True)
    parser.add_argument('--similarity', type=float, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    parser.add_argument('--objects', type=str, required=True)
    parser.add_argument('--cache_config_fname', type=str, required=True)
    parser.add_argument('--vg_objects', type=str, required=True)
    parser.add_argument('--vg_objects_anchors', type=str, required=True)
    args = parser.parse_args()
    patterns_fname = args.patterns
    unseen_objects_fname = args.unseen_objects
    with open(patterns_fname, 'rb') as f:
        patterns = pickle.load(f)
    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(args.vg_objects, args.vg_objects_anchors)
    config = Config(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence, args.objects, visual_config)
    unseen_objects = [line.strip() for line in fileinput.input(unseen_objects_fname)]

    htmls_lookup = scrape_htmls('htmls_unseen_objects.pkl', unseen_objects)

    # TODO coreferences
    logger.info(f'Using coreference: {config.coreference}')
    tuples = process_objects(unseen_objects, htmls_lookup, config)

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

    # Print tuples
    extracted_tuples = list(candidate_tuples.keys())
    tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence,
                           reverse=True)
    for t in tuples_sorted:
        logger.info(t.sentence)
        logger.info(f"{t.e1} {t.e2}")
        logger.info(t.confidence)
        logger.info("\n")

        # TODO can be sped up if necessary:
        #  https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html#sphx-glr-auto-examples-tutorials-run-annoy-py
        ms = config.word2vec.most_similar(positive=[t.e1], topn=5)
        logger.info(ms)


    # TODO actively search for objects lower in the wordtree hierarchy
    # TODO use embeddings to find similar objects in already found objects
    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
