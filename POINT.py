import fileinput
import json
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame

from breds.breds_inference import gather_sizes_with_bootstrapping_patterns, predict_using_tuples, load_patterns

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    patterns = load_patterns(cfg)

    # unseen_objects_fname = cfg.path.unseen_objects
    input: DataFrame = pd.from_csv(cfg.path.dev)
    unseen_objects = list(input['object'])




    # TODO check whether the objects aren't in the bootstrapped objects

    # TODO recognize abstract words and reject

    # TODO implement caching different patterns for visual and non-visual to enable comparison
    # Same for coreference. now it's just using the same patterns
    # BOOTSTRAP PATTERNS GENERATED USING VISUALS
    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(cfg, patterns, unseen_objects)

    point_predictions = predict_using_tuples(tuples_bootstrap, unseen_objects)

    # TODO think about format to save in
    # TODO should I just predict the maximum value? As we are looking for the maximum dimension of an object
    with open(f'point_predictions_bootstrapping_visual={cfg.visual_confidence}_coref={cfg.coreference}.json', 'w') as f:
        json.dump(point_predictions, f)

    res = []
    for _, row in input.iterrows():
        min = row['min']
        max = row['max']
        pred_size = point_predictions[row['object']]
        if pred_size is None:
            res.append(None)
        else:
            res.append(pred_size < max and pred_size > min)

    res = np.array(res)
    nan_count = np.isnan(res).sum()
    logger.info(f'Number of nans: {nan_count}')
    logger.info(f'Recall: {1 - (nan_count/len(res))}')

    res_clean = res[~np.isnan(res)]
    precision = np.mean(res_clean)
    logger.info(f'Precision: {precision}')

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
