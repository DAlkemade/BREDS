import fileinput
import json
import logging
import os
import pickle
from argparse import ArgumentParser
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

    patterns_fname = cfg.path.patterns_cache
    unseen_objects_fname = cfg.path.unseen_objects
    with open(patterns_fname, 'rb') as f:
        patterns = pickle.load(f)
    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)
    unseen_objects = set([line.strip() for line in fileinput.input(unseen_objects_fname)])

    #TODO recognize abstract words and reject
    word2vec_model = load_word2vec(cfg.parameters.word2vec_path)
    similar_words = find_similar_words(word2vec_model, unseen_objects)

    #TODO also add linguistics thing with removing head nouns

    # Create object lookup
    objects_lookup = create_reverse_lookup(similar_words)

    all_new_objects = set(objects_lookup.keys()).union(unseen_objects)

    #TODo extract html rettrieval from gather_sizes_with_bootstrapping_patterns so that I can also use them for regex

    # BOOTSTRAP PATTERNS GENERATED WITHOUT USING VISUALS

    # BOOTSTRAP PATTERNS GENERATED USING VISUALS
    # TODO replace this by loading cache
    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(config, patterns, all_new_objects)


    ##################### INFERENCE ###############################
    # REGEX

    # RAW NUMERIC BOOSTRAPPING

    # NUMERIC BOOSTRAPPING WITH VISUALS
    all_sizes = compile_results(tuples_bootstrap, objects_lookup, similar_words, unseen_objects)
    with open('backoff_sizes.json', 'w') as f:
        json.dump(all_sizes, f)
    # NUMERIC BOOTSTRAPPING WITH VISUALS AND BACKOFF

    # VISUAL PROPAGATION WITH RANGES
    # TODO maybe also do backoff for this

    # TODO maybe also use visual propagation here
    # TODO move this to other repo
    predict_sizes(all_sizes)

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
