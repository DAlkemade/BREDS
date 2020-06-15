import logging
import os
import pickle

import numpy as np
import pandas as pd

import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from learning_sizes_evaluation.monte_carlo_permutation_test import permutation_test
from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import comparison_dev_set

set_up_root_logger(f'COMBINE', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)

def get_result(golds, preds, tag, notes):
    coverage, selectivity = coverage_accuracy_relational(golds, preds, notes)
    logger.info(f'Coverage: {coverage}')
    logger.info(f'selectivity: {selectivity}')

    return RelationalResult(tag, selectivity, coverage)

def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    # TODO use config for this
    with open("bootstrapping_comparison_predictions_['direct', 'regex', 'median size'].pkl", 'rb') as f:
        linguistic_preds = list(pickle.load(f))
        logger.info(f'linguistics preds: {linguistic_preds}')
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
    notes_combined = list()
    for i, pair in enumerate(test_pairs):
        pred_visual, fraction_larger, note_visual = visual_preds[i]
        pred_linguistic, difference, note_linguistic = linguistic_preds[i]
        note = f'pair: {pair.e1} {pair.e2}'

        if pred_linguistic is None and pred_visual is None:
            pred = None
        else:
            if pred_linguistic is None:
                pred = pred_visual
                note += f'no linguistic, using visual: {note_visual}'
            elif pred_visual is None:
                pred = pred_linguistic
                note += f'no visual, using linguistic: {note_linguistic}'
            else:
                fraction_larger = abs(fraction_larger-.5)
                difference = abs(difference)
                if difference == 0.:
                    difference = 0.000000000001

                conf_visual = visual_conf_model.predict(np.reshape([fraction_larger], (-1, 1)))[0]
                conf_linguistic = linguistic_conf_model.predict(np.log10(np.reshape([difference], (-1,1))))[0]
                if difference > 100:
                    conf_visual = .93 #TODO get exact value
                if conf_visual > conf_linguistic:
                    pred = visual_preds[i][0]
                    visual += 1
                    note += f'visual conf ({conf_visual}) is higher than linguistic conf ({conf_linguistic}): {note_visual}'
                else:
                    pred = linguistic_preds[i][0]
                    linguistic += 1
                    note += f'linguistic conf ({conf_linguistic}) is higher than visual conf ({conf_visual}) : {note_linguistic}'
        preds_combine.append(pred)
        notes_combined.append(note)


    results = list()
    results.append(get_result(golds, preds_combine, 'combine', notes_combined))
    results.append(get_result(golds, [x[0] for x in linguistic_preds], 'linguistic', [x[2] for x in linguistic_preds]))
    results.append(get_result(golds, [x[0] for x in visual_preds], 'visual', [x[2] for x in visual_preds]))
    logger.info(f'Visual usages: {visual}')
    logger.info(f'Linguistic usages: {linguistic}')

    results_df = pd.DataFrame(results)
    results_df.to_csv('combine_results.csv')


    p = permutation_test([x[0] for x in linguistic_preds], preds_combine)
    logger.info(f'p-value {p} between combine and linguistic')







if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise