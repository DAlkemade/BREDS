import fileinput
import json
import logging
import math
import os
import pickle
import random
from datetime import datetime
from typing import List, Dict
import pandas as pd

import tqdm
import yaml
from box import Box
import numpy as np
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.config import Config
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from learning_sizes_evaluation.evaluate import precision_recall, range_distance

set_up_root_logger(f'RANGES_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def iterativily_find_size(lower_bounds_sizes, upper_bounds_sizes):
    l = lower_bounds_sizes.copy()
    u = upper_bounds_sizes.copy()
    total_objects = len(u) + len(l)
    size_scale = len(l) / total_objects
    logger.info(f'Scale: {size_scale}')
    count_l = 0
    count_r = 0
    while max(l) > min(u):
        r = random.random()
        if r < size_scale:
            l.remove(max(l))
            count_l += 1
        else:
            u.remove(min(u))
            count_r += 1
    logger.info(f'Removed total {count_l + count_r} out of {total_objects}; lower: {count_l} upper: {count_r}')
    logger.info(f'Final max(l): {max(l)} min(u): {min(u)}')
    return (max(l) + min(u)) / 2



def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)

    input: DataFrame = pd.read_csv(cfg.path.dev)
    input = input.astype({'object': str})
    unseen_objects = list(input['object'])
    logger.info(f'Unseen objects: {unseen_objects}')

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    logger.info(f'Objects: {objects}')
    G = build_cooccurrence_graph(objects, visual_config)

    with open(cfg.path.final_seeds_cache) as f:
        numeric_seeds = json.load(f)

    numeric_seeds = dict((key.strip().replace(' ', '_'), value) for (key, value) in numeric_seeds.items())

    point_predictions = dict()
    prop = VisualPropagation(G, config.visual_config)
    for unseen_object in unseen_objects:
        logger.info(f'Processing {unseen_object}')
        if unseen_object not in objects:
            logger.info(f'{unseen_object} not in visuals')
            continue
        none_count = 0
        lower_bounds = set()
        upper_bounds = set()
        for numeric_seed in tqdm.tqdm(numeric_seeds.keys()):
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
                logger.debug(f'{pair.e1} or {pair.e2} not in VG. Objects: {objects}')


        lower_bounds_sizes = fill_sizes_list(lower_bounds, numeric_seeds)
        upper_bounds_sizes = fill_sizes_list(upper_bounds, numeric_seeds)

        # size = predict_size_with_bounds(lower_bounds_sizes, upper_bounds_sizes)
        size = iterativily_find_size(lower_bounds_sizes, upper_bounds_sizes)
        point_predictions[unseen_object] = size
        logger.info(f'\nObject: {unseen_object}')
        logger.info(f'Size: {size}')
        logger.info(f"None count: {none_count} out of {len(numeric_seeds.keys())}")
        logger.info(f"Lower bounds (n={len(lower_bounds)}): mean: {np.mean(lower_bounds_sizes)} median: {np.median(lower_bounds_sizes)}\n\t{lower_bounds}\n\t{lower_bounds_sizes}")
        logger.info(f"Upper bounds (n={len(upper_bounds)}): mean: {np.mean(upper_bounds_sizes)} median: {np.median(upper_bounds_sizes)}\n\t{upper_bounds}\n\t{upper_bounds_sizes}")

    with open(f'point_predictions_visual_ranges.pkl', 'wb') as f:
        pickle.dump(point_predictions, f)

    precision_recall(input, point_predictions)
    range_distance(input, point_predictions)

    logger.info('Finished')

def predict_size_with_bounds(lower_bounds_sizes, upper_bounds_sizes) -> float:
    data = list(zip(lower_bounds_sizes, len(lower_bounds_sizes) * [0])) + list(
        zip(upper_bounds_sizes, len(upper_bounds_sizes) * [1]))
    x, y = zip(*data)
    x = np.reshape(x, (-1, 1))


    pivot_point = get_boundary_value(x, y)
    return pivot_point


def get_boundary_value(x, y):
    clf = SVC(gamma='auto', decision_function_shape='ovo', kernel='linear', C=100000000, cache_size=2000)


    # logger.info(f'scaler: scale {scaler.scale_} mean {scaler.mean_}')
    logger.info(f'Fit svm on {len(y)} data points')
    #TODO maybe cluster with kemans to reduce number of data points if it gets stuck
    clf.fit(x, y)
    #w_norm is a in y = a*x + b
    intercept = clf.intercept_
    w_norm = np.linalg.norm(clf.coef_)
    scale = np.reshape(np.linspace(0, 10, 500), (-1, 1))
    boundary = clf.decision_function(scale)
    dist = boundary / w_norm
    plt.plot(scale, dist)
    plt.show()
    #return the distance of size 0 to the boundary, which is the size
    z = np.reshape([0.], (-1, 1))

    pivot_point = -1 * clf.decision_function(z)[0] / w_norm

    return pivot_point


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
