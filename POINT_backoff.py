import logging
import os
from typing import List

from datetime import datetime

import pandas as pd
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import precision_recall, range_distance, Result, distances_hist, get_distances
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame

from breds.breds_inference import predict_sizes, load_patterns, BackoffSettings, get_all_sizes_bootstrapping, \
    calc_median

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    # TODO also add linguistics thing with removing head nouns

    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Bothrex(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    patterns = load_patterns(cfg)

    # TODO check whether the objects aren't in the bootstrapped objects
    input_fname = cfg.path.dev
    input: DataFrame = pd.read_csv(input_fname)
    input = input.astype({'object': str})
    unseen_objects = list(input['object'])
    logger.info(f'Unseen objects: {unseen_objects}')

    cache_fname = 'backoff_sizes.pkl'
    all_sizes = get_all_sizes_bootstrapping(cache_fname, cfg, input_fname, patterns, unseen_objects)
    median = calc_median(cfg)

    # with open(f'backoff_predictions.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)
    results = list()
    settings: List[BackoffSettings] = [
        BackoffSettings(use_direct=True),
        BackoffSettings(use_word2vec=True),
        BackoffSettings(use_hypernyms=True),
        BackoffSettings(use_hyponyms=True),
        BackoffSettings(use_head_noun=True),
        BackoffSettings(use_median_size=True),
        BackoffSettings(use_direct=True, use_word2vec=True),
        BackoffSettings(use_direct=True, use_hypernyms=True),
        BackoffSettings(use_direct=True, use_hyponyms=True),
        BackoffSettings(use_direct=True, use_head_noun=True),
        BackoffSettings(use_direct=True, use_hyponyms=True),
        BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True),
        BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True)
    ]
    for setting in settings:
        logger.info(f'Setting: {setting.print()}')
        results.append(evaluate_settings(setting, all_sizes, unseen_objects, input, median))
    results_df = pd.DataFrame(results)
    results_df.to_csv('results_backoff.csv')

    word2vecsettings = BackoffSettings(use_word2vec=True)
    word2vecpreds = predict_sizes(all_sizes, unseen_objects, word2vecsettings, median)
    distances_hist({'word2vec fallback': get_distances(input, word2vecpreds)}, ['word2vec fallback'], save=True)
    logger.info('Finished')


def evaluate_settings(settings:BackoffSettings, all_sizes, objects, input, median):
    settings.print()
    predictions = predict_sizes(all_sizes, objects, settings, median)
    selectivity, coverage = precision_recall(input, predictions)
    mean, mean_squared, median = range_distance(input, predictions)
    return Result(settings.print(), selectivity, coverage, mean, mean_squared, median)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
