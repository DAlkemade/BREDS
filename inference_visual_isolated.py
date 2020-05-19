import fileinput
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List

from logging_setup_dla.logging import set_up_root_logger
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.config import Config

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--test_pairs', type=str, required=True)
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
    test_pairs_fname = args.test_pairs

    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(args.vg_objects, args.vg_objects_anchors)
    config = Config(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence,
                    args.objects, visual_config)

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    logger.info(f'Objects: {objects}')
    G = build_cooccurrence_graph(objects, visual_config)

    test_pairs: List[Pair] = list()
    for line in fileinput.input(test_pairs_fname):
        split = line.split(',')
        test_pairs.append(Pair(split[0].strip().replace(' ', '_'), split[1].strip().replace(' ', '_')))

    prop = VisualPropagation(G, config.visual_config)
    for test_pair in test_pairs:
        if test_pair.both_in_list(objects):
            # TODO: bigrams not found
            fraction_larger = prop.compare_pair(test_pair)
            logger.info(f'{test_pair.e1} {test_pair.e2} fraction larger: {fraction_larger}')
        else:
            logger.warning(f'{test_pair.e1} or {test_pair.e2} not in VG. Objects: {objects}')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
