import fileinput
import logging
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import predict_sizes, gather_sizes_with_bootstrapping_patterns, compile_results, \
    find_similar_words, create_reverse_lookup
from breds.config import Config
from visual_size_comparison.config import VisualConfig

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


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
    config = Config(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence,
                    args.objects, visual_config)
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
