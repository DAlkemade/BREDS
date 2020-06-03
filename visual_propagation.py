import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import tqdm
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from logging_setup_dla.logging import set_up_root_logger
from matplotlib import pyplot as plt
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.breds_inference import find_similar_words, BackoffSettings
from breds.config import Config, load_word2vec

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def compare_linguistic_with_backoff(setting, similar_words, test_pair):
    #TODO
    pass


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        # cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    test_pairs, unseen_objects = comparison_dev_set(cfg)

    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(cfg.path.vg_objects, cfg.path.vg_objects_anchors)
    config = Config(cfg, visual_config)

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    logger.info(f'Objects: {objects}')
    G = build_cooccurrence_graph(objects, visual_config)

    word2vec_model = load_word2vec(cfg.parameters.word2vec_path)
    similar_words = find_similar_words(word2vec_model, unseen_objects, n_word2vec=200)

    # calc coverage and precision
    results = list()
    settings: List[BackoffSettings] = [
        BackoffSettings(use_direct=True),
        BackoffSettings(use_word2vec=True),
        BackoffSettings(use_hypernyms=True),
        BackoffSettings(use_hyponyms=True),
        BackoffSettings(use_head_noun=True),
        # BackoffSettings(use_direct=True, use_word2vec=True),
        # BackoffSettings(use_direct=True, use_hypernyms=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True),
        # BackoffSettings(use_direct=True, use_head_noun=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True)
    ]
    golds = [p.larger for p in test_pairs]

    for setting in settings:
        preds = list()
        not_recognized_count = 0

        prop = VisualPropagation(G, config.visual_config)
        logger.info(f'\nRunning for setting {setting.print()}')

        for test_pair in tqdm.tqdm(test_pairs):
            #TODO let both return confidence; use the higher one
            res_visual = compare_visual_with_backoff(objects, prop, setting, similar_words, test_pair)
            res_textual = compare_linguistic_with_backoff(setting, similar_words, test_pair)

            preds.append(res_visual)

        useful_counts = prop.useful_path_counts
        plt.hist(useful_counts, bins=1000)
        plt.xlabel('Number of useful paths')
        plt.savefig('useful_paths.png')

        useful_counts = np.array(useful_counts)
        logger.info(f'Number of objects with no useful path: {len(np.extract(useful_counts == 0, useful_counts))}')
        logger.info(f'Not recog count: {not_recognized_count}')

        logger.info(f'Total number of test cases: {len(golds)}')
        coverage, selectivity = coverage_accuracy_relational(golds, preds)
        logger.info(f'Coverage: {coverage}')
        logger.info(f'selectivity: {selectivity}')

        results.append(RelationalResult(setting.print(), selectivity, coverage))

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_visual_backoff.csv')


def compare_visual_with_backoff(objects, prop, setting, similar_words, test_pair):
    object1 = test_pair.e1.replace('_', ' ')
    object2 = test_pair.e2.replace('_', ' ')
    # TODO implement backoff mechanism
    recognizable_objects1 = fill_objects_list(object1, setting, objects, similar_words)
    recognizable_objects2 = fill_objects_list(object2, setting, objects, similar_words)
    comparisons = list()
    for o1 in recognizable_objects1:
        for o2 in recognizable_objects2:
            fraction_larger = prop.compare_pair(Pair(o1, o2))
            if fraction_larger is not None:
                comparisons.append(fraction_larger)
            logger.debug(f'{o1} {o2} fraction larger: {fraction_larger}')
    if len(comparisons) > 0:
        # larger_results = [c > .5 for c in comparisons]
        # fraction_larger_mean = np.mean(larger_results)
        fraction_larger_mean = np.mean(comparisons)
        res = fraction_larger_mean > .5
    else:
        res = None
    return res


def comparison_dev_set(cfg):
    input: pd.DataFrame = pd.read_csv(cfg.path.dev)
    input = input.astype({'object': str})
    input.set_index(['object'], inplace=True, drop=False)
    unseen_objects = list(input['object'])
    logger.info(f'Unseen objects: {unseen_objects}')
    test_pairs: List[Pair] = list()
    row_count = len(input.index)
    for i in range(row_count):
        for j in range(i + 1, row_count):
            row1 = input.iloc[i]
            row2 = input.iloc[j]
            pair = Pair(row1.at['object'].strip().replace(' ', '_'), row2.at['object'].strip().replace(' ', '_'))

            larger1 = float(row1.at['min']) > float(row2.at['max'])
            larger2 = float(row2.at['min']) > float(row1.at['max'])
            if not larger1 and not larger2:
                # ranges overlap, not evaluating
                continue
            if larger1:
                gold_larger = True
            else:
                gold_larger = False
            pair.larger = gold_larger

            test_pairs.append(pair)
    return test_pairs, unseen_objects


def check_if_in_vg(word_list, vg_objects):
    res = list()
    for word in word_list:
        word = word.replace(' ', "_")
        if word in vg_objects:
            res.append(word)
    return res


def fill_objects_list(entity: str, setting: BackoffSettings, vg_objects: list, similar_words_lookup):
    synset_string = entity.replace(' ','_')
    entity_string = entity.replace('_', " ")
    recognizable_objects = list()
    similar_words = similar_words_lookup[entity_string]
    #TODO check if this gets two-word objects!!
    if setting.use_direct:
        if synset_string in vg_objects:
            recognizable_objects.append(synset_string)

    elif len(recognizable_objects) ==0 and setting.use_hyponyms:
        hyponyms = similar_words['hyponyms']
        if len(hyponyms) > 0:
            recognizable_objects += check_if_in_vg(hyponyms, vg_objects)

    elif len(recognizable_objects) == 0 and setting.use_hypernyms:
        hypernyms = similar_words['hypernyms']
        if len(hypernyms) > 0:
            recognizable_objects += check_if_in_vg(hypernyms, vg_objects)

    elif len(recognizable_objects) == 0 and setting.use_head_noun:
        head_nouns = similar_words['head_noun']
        assert type(head_nouns) is list
        if len(head_nouns) > 0:
            recognizable_objects += check_if_in_vg(head_nouns, vg_objects)

    elif len(recognizable_objects) ==0 and setting.use_word2vec:
        word2vecs = similar_words['word2vec']
        if len(word2vecs) > 0:
            all_word2vecs_in_vg = check_if_in_vg(word2vecs, vg_objects)
            if len(all_word2vecs_in_vg) > 0:
                best_three = all_word2vecs_in_vg[:min(3,len(all_word2vecs_in_vg))]
                recognizable_objects += best_three

    else:
        logger.debug(f'{synset_string} and fallback objects not in VG.')
    return recognizable_objects

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
