import fileinput
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
from breds.config import Config
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


    cache_fname = 'inference_cache.pkl'
    similar_words = find_similar_words(config, unseen_objects)

    # Create object lookup
    objects_lookup = create_reverse_lookup(similar_words)

    all_new_objects = set(objects_lookup.keys()).union(unseen_objects)

    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(cache_fname, config, patterns, all_new_objects)

    all_sizes = compile_results(tuples_bootstrap, objects_lookup, similar_words, unseen_objects)

    # TODO maybe also use visual propagation here

    predict_sizes(all_sizes)

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
