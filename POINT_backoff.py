import json
import logging
import os
from datetime import datetime

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import predict_sizes, gather_sizes_with_bootstrapping_patterns, compile_results, \
    find_similar_words, create_reverse_lookup, load_patterns, load_unseen_objects
from breds.config import load_word2vec

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    # TODO also add linguistics thing with removing head nouns
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    patterns = load_patterns(cfg)

    # TODO check whether the objects aren't in the bootstrapped objects

    unseen_objects = load_unseen_objects(cfg)

    word2vec_model = load_word2vec(cfg.parameters.word2vec_path)
    similar_words = find_similar_words(word2vec_model, unseen_objects)

    # Create object lookup
    objects_lookup = create_reverse_lookup(similar_words)

    all_new_objects = set(objects_lookup.keys()).union(unseen_objects)

    # TODo extract html rettrieval from gather_sizes_with_bootstrapping_patterns so that I can also use them for regex

    # BOOTSTRAP PATTERNS GENERATED WITHOUT USING VISUALS

    # BOOTSTRAP PATTERNS GENERATED USING VISUALS
    # TODO replace this by loading cache
    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(cfg, patterns, all_new_objects)

    all_sizes = compile_results(tuples_bootstrap, objects_lookup, similar_words, unseen_objects)
    with open('backoff_sizes.json', 'w') as f:
        json.dump(all_sizes, f)

    # TODO move this to other repo
    predict_sizes(all_sizes)

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
