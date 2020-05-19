import fileinput
import json
import logging
import os
from datetime import datetime
from typing import List

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.config import Config

set_up_root_logger(f'RANGES_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg.path.configuration, cfg.path.seeds_file, cfg.path.negative_seeds, cfg.parameters.similarity,
                    cfg.parameters.confidence,
                    cfg.path.objects, visual_config)

    test_objects_fname = cfg.path.unseen_objects

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    logger.info(f'Objects: {objects}')
    G = build_cooccurrence_graph(objects, visual_config)

    unseen_objects: List[str] = list()
    for line in fileinput.input(test_objects_fname):
        unseen_objects.append(line.strip().replace(' ', '_'))

    with open(cfg.path.final_seeds_cache) as f:
        numeric_seeds = json.load(f)

    numeric_seeds = dict((key.strip().replace(' ', '_'), value) for (key, value) in numeric_seeds.items())


    prop = VisualPropagation(G, config.visual_config)
    for unseen_object in unseen_objects:
        none_count = 0
        for numeric_seed in numeric_seeds.keys():
            pair = Pair(unseen_object, numeric_seed)
            if pair.both_in_list(objects):
                fraction_larger = prop.compare_pair(pair)
                if fraction_larger is None:
                    none_count += 1
                logger.info(f'{pair.e1} {pair.e2} fraction larger: {fraction_larger}')
            else:
                logger.warning(f'{pair.e1} or {pair.e2} not in VG. Objects: {objects}')
        logger.info(f"None count for {unseen_object}: {none_count} out of {len(numeric_seeds.keys())}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
