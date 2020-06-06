import logging
import os
import pickle
from datetime import datetime
from math import floor, ceil
from typing import List

import pandas as pd
import tqdm
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from logging_setup_dla.logging import set_up_root_logger
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from visual_size_comparison.propagation import Pair
from size_comparisons.scraping.lengths_regex import parse_documents_for_lengths, predict_size_regex

from breds.breds_inference import BackoffSettings, comparison_dev_set, get_all_sizes_bootstrapping, \
    load_patterns, predict_size, calc_median
from breds.htmls import scrape_htmls
import numpy as np
from matplotlib import pyplot as plt

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def compare_linguistic_with_backoff(setting: BackoffSettings, all_sizes, test_pair: Pair, median: float, regex_predictions) -> (bool, float):
    #TODO think of a proxy for confidence using the backoff level and the difference between the sizes
    o1 = test_pair.e1.replace('_', ' ')
    o2 = test_pair.e2.replace('_', ' ')
    regex1 = regex_predictions[o1]
    regex2 = regex_predictions[o2]
    res1 = predict_size(all_sizes, setting, o1, median_size=median, regex_size=regex1)
    res2 = predict_size(all_sizes, setting, o2, median_size=median, regex_size=regex2)
    if res1 is not None and res2 is not None:
        diff = res1 - res2
        res = diff > 0
    else:
        diff = None
        res = None

    return res, diff


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    test_pairs, unseen_objects = comparison_dev_set(cfg)
    unseen_objects = [o.replace('_', " ") for o in unseen_objects]

    # TODO check whether the objects aren't in the bootstrapped objects

    patterns = load_patterns(cfg)
    median = calc_median(cfg)
    logger.info(f'Median: {median}')
    cache_fname = 'backoff_sizes.pkl'
    input_fname = cfg.path.dev
    settings: List[BackoffSettings] = [
        # BackoffSettings(use_direct=True),
        # BackoffSettings(use_regex=True),
        # BackoffSettings(use_direct=True, use_regex=True),
        # BackoffSettings(use_direct=True, use_word2vec=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True, use_median_size=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_word2vec=True, use_regex=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True, use_hypernyms=True, use_regex=True),
        BackoffSettings(use_direct=True, use_regex=True, use_median_size=True),
    ]
    word2vec_needed = False
    for setting in settings:
        if setting.use_word2vec:
            word2vec_needed = True
    all_sizes = get_all_sizes_bootstrapping(cache_fname, cfg, input_fname, patterns, unseen_objects, use_word2vec=word2vec_needed)
    logger.info(f'all_sizes: {all_sizes}')

    htmls_lookup = scrape_htmls(cfg.path.htmls_cache, unseen_objects)
    sizes_regex, _ = parse_documents_for_lengths(unseen_objects, htmls_lookup)
    regex_predictions = dict()
    for o in unseen_objects:
        mean = predict_size_regex(o, sizes_regex)
        regex_predictions[o] = mean

    # calc coverage and precision
    results = list()

    golds = [p.larger for p in test_pairs]
    for setting in settings:
        preds = list()
        diffs = list()

        for test_pair in tqdm.tqdm(test_pairs):
            #TODO return confidence; use the higher one
            res, diff = compare_linguistic_with_backoff(setting, all_sizes, test_pair, median, regex_predictions)
            diffs.append(diff)
            preds.append(res)

        with open(f'bootstrapping_comparison_predictions_{setting.print()}.pkl', 'wb') as f:
            pickle.dump(zip(preds, diffs), f)

        logger.info(f'Total number of test cases: {len(golds)}')
        coverage, selectivity = coverage_accuracy_relational(golds, preds)
        logger.info(f'Coverage: {coverage}')
        logger.info(f'selectivity: {selectivity}')

        results.append(RelationalResult(setting.print(), selectivity, coverage))


        assert len(diffs) == len(preds)
        corrects_not_none = list()
        diffs_not_none = list()
        for i, diff in enumerate(diffs):
            gold = golds[i]
            res = preds[i]
            if diff is not None and diff != 0:
                corrects_not_none.append(gold == res)
                diffs_not_none.append(abs(diff))
        #TODO do something special for when diff == 0
        regr = Ridge(alpha=1.0)
        regr.fit(np.reshape(np.log10(diffs_not_none), (-1, 1)), corrects_not_none)
        poly_ridge = make_pipeline(PolynomialFeatures(2), Ridge())
        poly_ridge.fit(np.reshape(np.log10(diffs_not_none), (-1, 1)), corrects_not_none)
        svm = make_pipeline(StandardScaler(), SVR(kernel='poly'))
        svm.fit(np.reshape(np.log10(diffs_not_none), (-1, 1)), corrects_not_none)
        svm2 = make_pipeline(StandardScaler(), SVR(kernel='poly', C=10))
        svm2.fit(np.reshape(np.log10(diffs_not_none), (-1, 1)), corrects_not_none)
        # x = np.linspace(0, 10000, 1000)
        # plt.savefig('test_svm.png')

        minimum_power = floor(np.log(min(diffs_not_none)))
        maximum_power = ceil(np.log(max(diffs_not_none)))
        bin_means, bin_edges, binnumber = stats.binned_statistic(diffs_not_none, corrects_not_none, 'mean', bins=np.logspace(minimum_power, maximum_power, 20))
        fig, ax = plt.subplots()
        plt.plot(diffs_not_none, corrects_not_none, 'b.', label='raw data')
        plt.plot(diffs_not_none, regr.predict(np.reshape(np.log10(diffs_not_none), (-1,1))), label='linear ridge prediction')
        plt.plot(diffs_not_none, poly_ridge.predict(np.reshape(np.log10(diffs_not_none), (-1, 1))),
                 label='poly ridge prediction')
        plt.plot(diffs_not_none, svm.predict(np.reshape(np.log10(diffs_not_none), (-1, 1))), label='svm prediction')
        plt.plot(diffs_not_none, svm2.predict(np.reshape(np.log10(diffs_not_none), (-1, 1))), label='svm 2 prediction')

        plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
                   label='binned statistic of data')
        plt.legend()
        plt.xlabel('Absolute difference in size')
        plt.ylabel('Selectivity')
        ax.set_xscale('log')
        plt.savefig('differences.png')
        plt.show()

        correlation, _ = pearsonr(diffs_not_none, corrects_not_none)
        logger.info(f'Pearsons correlation: {correlation}')

        correlation_spearman, _ = spearmanr(np.array(diffs_not_none), b=np.array(corrects_not_none))
        logger.info(f'Spearman correlation: {correlation_spearman}')




    results_df = pd.DataFrame(results)
    results_df.to_csv('results_bootstrapping_comparison_backoff.csv')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
