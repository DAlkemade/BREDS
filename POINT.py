import fileinput
import json
import logging
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import predict_sizes, gather_sizes_with_bootstrapping_patterns, compile_results, \
    find_similar_words, create_reverse_lookup
from breds.config import Config, load_word2vec
from visual_size_comparison.config import VisualConfig

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    patterns_paths = cfg.path.patterns
    if cfg.parameters.coreference:
        patterns_paths = patterns_paths.coref
    else:
        patterns_paths = patterns_paths.coref
    if cfg.parameters.visual_confidence:
        patterns_fname = patterns_paths.visual
    else:
        patterns_fname = patterns_paths.no_visual
    unseen_objects_fname = cfg.path.unseen_objects
    with open(patterns_fname, 'rb') as f:
        patterns = pickle.load(f)
    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)
    unseen_objects = set([line.strip() for line in fileinput.input(unseen_objects_fname)])

    #TODO recognize abstract words and reject


    # TODO implement caching different patterns for visual and non-visual to enable comparison
    # Same for coreference. now it's just using the same patterns
    # BOOTSTRAP PATTERNS GENERATED USING VISUALS
    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(config, patterns, unseen_objects)
    for t in tuples_bootstrap.keys():
        logger.info(t.sentence)
        logger.info(f"{t.e1} {t.e2}")
        logger.info(t.confidence)
        logger.info("\n")


    collated = defaultdict(list)
    for t in tuples_bootstrap:
        collated[t.e1].append(t.e2)

    point_predictions = dict()
    for o in unseen_objects:
        point_predictions[o] = max(collated[o])

    # TODO think about format to save in
    # TODO should I just predict the maximum value? As we are looking for the maximum dimension of an object
    with open(f'point_predictions_bootstrapping_visual={config.visual}_coref={config.coreference}.json', 'w') as f:
        json.dump(point_predictions, f)
    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
