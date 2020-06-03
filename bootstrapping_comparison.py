import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
import tqdm
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from logging_setup_dla.logging import set_up_root_logger
from visual_size_comparison.propagation import Pair

from breds.breds_inference import BackoffSettings, comparison_dev_set, get_all_sizes_bootstrapping, \
    load_patterns, predict_size, calc_median

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def compare_linguistic_with_backoff(setting: BackoffSettings, all_sizes, test_pair: Pair, median: float) -> bool:
    #TODO think of a proxy for confidence using the backoff level and the difference between the sizes
    res1 = predict_size(all_sizes, setting, test_pair.e1, median_size=median)
    res2 = predict_size(all_sizes, setting, test_pair.e2, median_size=median)
    if res1 is not None and res2 is not None:
        res = res1 > res2
    else:
        res = None

    return res


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    test_pairs, unseen_objects = comparison_dev_set(cfg)

    # TODO check whether the objects aren't in the bootstrapped objects

    patterns = load_patterns(cfg)
    median = calc_median(cfg)
    logger.info(f'Median: {median}')
    cache_fname = 'backoff_sizes.pkl'
    input_fname = cfg.path.dev

    all_sizes = get_all_sizes_bootstrapping(cache_fname, cfg, input_fname, patterns, unseen_objects)

    # calc coverage and precision
    results = list()
    settings: List[BackoffSettings] = [
        BackoffSettings(use_direct=True),
        # BackoffSettings(use_word2vec=True),
        # BackoffSettings(use_hypernyms=True),
        # BackoffSettings(use_hyponyms=True),
        # BackoffSettings(use_head_noun=True),
        # BackoffSettings(use_direct=True, use_word2vec=True),
        # BackoffSettings(use_direct=True, use_hypernyms=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True),
        # BackoffSettings(use_direct=True, use_head_noun=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True)
        BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True),
        BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True, use_median_size=True)
    ]
    golds = [p.larger for p in test_pairs]

    for setting in settings:
        preds = list()


        for test_pair in tqdm.tqdm(test_pairs):
            #TODO return confidence; use the higher one
            res = compare_linguistic_with_backoff(setting, all_sizes, test_pair, median)
            preds.append(res)


        logger.info(f'Total number of test cases: {len(golds)}')
        coverage, selectivity = coverage_accuracy_relational(golds, preds)
        logger.info(f'Coverage: {coverage}')
        logger.info(f'selectivity: {selectivity}')

        results.append(RelationalResult(setting.print(), selectivity, coverage))

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_bootstrapping_comparison_backoff.csv')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
