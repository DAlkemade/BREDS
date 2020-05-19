import fileinput
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.config import Config

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)

def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)
    test_pairs_fname = cfg.path.test_pairs

    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg.path.configuration, cfg.path.seeds_file, cfg.path.negative_seeds, cfg.parameters.similarity,
                    cfg.parameters.confidence,
                    cfg.path.objects, visual_config)

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
