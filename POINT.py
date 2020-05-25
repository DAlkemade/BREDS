import fileinput
import json
import logging
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame
from learning_sizes_evaluation.evaluate import precision_recall, range_distance

from breds.breds_inference import gather_sizes_with_bootstrapping_patterns, predict_using_tuples, load_patterns

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)
    os.chdir(cfg.path.work_dir)
    patterns = load_patterns(cfg)

    # unseen_objects_fname = cfg.path.unseen_objects
    input: DataFrame = pd.read_csv(cfg.path.dev)
    input = input.astype({'object': str})
    unseen_objects = list(input['object'])
    logger.info(f'Unseen objects: {unseen_objects}')

    # TODO check whether the objects aren't in the bootstrapped objects

    # TODO recognize abstract words and reject

    # TODO implement caching different patterns for visual and non-visual to enable comparison
    # Same for coreference. now it's just using the same patterns
    # BOOTSTRAP PATTERNS GENERATED USING VISUALS
    tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(cfg, patterns, unseen_objects)

    point_predictions = predict_using_tuples(tuples_bootstrap, unseen_objects, maximum=True)

    # TODO think about format to save in
    # TODO should I just predict the maximum value? As we are looking for the maximum dimension of an object
    with open(f'point_predictions_bootstrapping_visual={cfg.parameters.visual_confidence}_coref={cfg.parameters.coreference}dev_thr={cfg.parameters.dev_threshold}.pkl', 'wb') as f:
        pickle.dump(point_predictions, f)

    precision_recall(input, point_predictions)
    range_distance(input, point_predictions)

    logger.info('Finished')





if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise