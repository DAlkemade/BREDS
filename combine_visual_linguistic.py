import logging
import os
import pickle

import numpy as np
import pandas as pd

import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import comparison_dev_set

set_up_root_logger(f'COMBINE', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)

def get_result(golds, preds, tag):
    coverage, selectivity = coverage_accuracy_relational(golds, preds)
    logger.info(f'Coverage: {coverage}')
    logger.info(f'selectivity: {selectivity}')

    return RelationalResult(tag, selectivity, coverage)

def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    # TODO use config for this
    with open("bootstrapping_comparison_predictions_['direct', 'regex', 'median size'].pkl", 'rb') as f:
        linguistic_preds = list(pickle.load(f))
    with open("visual_comparison_predictions_['direct', 'word2vec', 'hypernyms'].pkl", 'rb') as f:
        visual_preds = list(pickle.load(f))
    with open("bootstrapping_confidence_model.pkl", 'rb') as f:
        linguistic_conf_model = pickle.load(f)
    with open("visual_confidence_model.pkl", 'rb') as f:
        visual_conf_model = pickle.load(f)

    # confidences_linguistic = linguistic_conf_model.predict(np.log10(np.reshape([p[1] for p in linguistic_preds], (-1,1))))
    # confidences_visual = visual_conf_model.predict(np.reshape([p[1] for p in linguistic_preds], (-1, 1)))

    test_pairs, _ = comparison_dev_set(cfg)
    golds = [p.larger for p in test_pairs]

    preds_combine = list()
    linguistic = 0
    visual = 0
    for i, pair in enumerate(test_pairs):
        pred_linguistic, difference = linguistic_preds[i]
        pred_visual, fraction_larger = visual_preds[i]


        if pred_linguistic is None and pred_visual is None:
            pred = None
        else:
            if pred_linguistic is None:
                pred = pred_visual
            elif pred_visual is None:
                pred = pred_linguistic
            else:
                fraction_larger = abs(fraction_larger)
                difference = abs(difference)
                if difference == 0.:
                    difference = 0.000000000001

                conf_visual = visual_conf_model.predict(np.reshape([fraction_larger], (-1, 1)))[0]
                conf_linguistic = linguistic_conf_model.predict(np.log10(np.reshape([difference], (-1,1))))[0]
                if conf_visual > conf_linguistic:
                    pred = visual_preds[i][0]
                    visual += 1
                else:
                    pred = linguistic_preds[i][0]
                    linguistic += 1
        preds_combine.append(pred)


    results = list()
    results.append(get_result(golds, preds_combine, 'combine'))
    results.append(get_result(golds, linguistic_preds, 'linguistic'))
    results.append(get_result(golds, visual_preds, 'visual'))
    logger.info(f'Visual usages: {visual}')
    logger.info(f'Linguistic usages: {linguistic}')

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_bootstrapping_comparison_backoff.csv')







if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise