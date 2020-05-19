import fileinput
import json
import logging
import os
from datetime import datetime
from typing import List, Dict

import yaml
from box import Box
import numpy as np
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
    config = Config(cfg, visual_config)

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
        if unseen_object not in objects:
            logger.info(f'{unseen_object} not in visuals')
            continue
        none_count = 0
        lower_bounds = set()
        upper_bounds = set()
        for numeric_seed in numeric_seeds.keys():
            pair = Pair(unseen_object, numeric_seed)
            if pair.both_in_list(objects):
                fraction_larger = prop.compare_pair(pair)
                if fraction_larger is None:
                    none_count += 1
                    continue
                if fraction_larger < .5:
                    upper_bounds.add(numeric_seed)
                if fraction_larger > .5:
                    lower_bounds.add(numeric_seed)
                logger.debug(f'{pair.e1} {pair.e2} fraction larger: {fraction_larger}')
            else:
                logger.warning(f'{pair.e1} or {pair.e2} not in VG. Objects: {objects}')


        lower_bounds_sizes = fill_sizes_list(lower_bounds, numeric_seeds)
        upper_bounds_sizes = fill_sizes_list(upper_bounds, numeric_seeds)

        logger.info(f"None count for {unseen_object}: {none_count} out of {len(numeric_seeds.keys())}")
        logger.info(f"Lower bounds (n={len(lower_bounds)}): mean: {np.mean(lower_bounds_sizes)} median: {np.median(lower_bounds_sizes)}\n\t{lower_bounds}\n\t{lower_bounds_sizes}")
        logger.info(f"Upper bounds (n={len(upper_bounds)}): mean: {np.mean(upper_bounds_sizes)} median: {np.median(upper_bounds_sizes)}\n\t{upper_bounds}\n\t{upper_bounds_sizes}")


def fill_sizes_list(objects: set, seeds_dict: Dict[str, list]) -> list:
    res = list()
    for o in objects:
        sizes = seeds_dict[o]
        mean = np.mean(sizes)
        res.append(mean)
    res = list(sorted(res))
    return res


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
