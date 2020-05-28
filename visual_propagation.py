import fileinput
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import tqdm
import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from sklearn.metrics import precision_score, recall_score
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.config import Config
import pandas as pd
from learning_sizes_evaluation.evaluate import coverage_accuracy

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)

def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    input: pd.DataFrame = pd.read_csv(cfg.path.dev)
    input = input.astype({'object': str})
    input.set_index(['object'], inplace=True, drop=False)
    unseen_objects = list(input['object'])
    logger.info(f'Unseen objects: {unseen_objects}')
    test_pairs: List[Pair] = list()

    row_count = len(input.index)
    for i in range(row_count):
        for j in range(i+1,row_count):
            row1 = input.iloc[i]
            row2 = input.iloc[j]
            pair = Pair(row1.at['object'].strip().replace(' ', '_'), row2.at['object'].strip().replace(' ', '_'))
            test_pairs.append(pair)


    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    logger.info(f'Objects: {objects}')
    G = build_cooccurrence_graph(objects, visual_config)



    prop = VisualPropagation(G, config.visual_config)


    # calc coverage and precision
    golds = list()
    preds = list()
    for test_pair in tqdm.tqdm(test_pairs):
        object1 = test_pair.e1.replace('_', ' ')
        object2 = test_pair.e2.replace('_', ' ')
        row1 = input.loc[object1]
        row2 = input.loc[object2]

        larger1 = float(row1.at['min']) > float(row2.at['max'])
        larger2 = float(row2.at['min']) > float(row1.at['max'])
        if not larger1 and not larger2:
            # ranges overlap, not evaluating
            continue
        if larger1:
            gold_larger = True
        else:
            gold_larger = False
        golds.append(gold_larger)

        if test_pair.both_in_list(objects):
            fraction_larger = prop.compare_pair(test_pair)
            if fraction_larger is None:
                res = None
            else:
                res = fraction_larger > .5
            logger.debug(f'{test_pair.e1} {test_pair.e2} fraction larger: {fraction_larger}')
        else:
            res = None
            logger.debug(f'{test_pair.e1} or {test_pair.e2} not in VG. Objects: {objects}')

        preds.append(res)

    logger.info(f'Total number of test cases: {len(golds)}')
    coverage, accuracy = coverage_accuracy(golds, preds)
    logger.info(f'Coverage: {coverage}')
    logger.info(f'Accuracy: {accuracy}')



if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
