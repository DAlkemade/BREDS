import logging
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import tqdm
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import coverage_accuracy_relational, RelationalResult
from logging_setup_dla.logging import set_up_root_logger
from matplotlib import pyplot as plt, colors, cm
from matplotlib.scale import SymmetricalLogTransform
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from visual_size_comparison.config import VisualConfig
from visual_size_comparison.propagation import build_cooccurrence_graph, Pair, VisualPropagation

from breds.breds_inference import find_similar_words, BackoffSettings, comparison_dev_set
from breds.config import Config, load_word2vec

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


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
        # BackoffSettings(use_direct=True),
        # BackoffSettings(use_word2vec=True),
        # BackoffSettings(use_hypernyms=True),
        # BackoffSettings(use_hyponyms=True),
        # BackoffSettings(use_head_noun=True),
        # BackoffSettings(use_direct=True, use_word2vec=True),
        BackoffSettings(use_direct=True, use_word2vec=True, use_hypernyms=True),
        # BackoffSettings(use_direct=True, use_hypernyms=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True),
        # BackoffSettings(use_direct=True, use_head_noun=True),
        # BackoffSettings(use_direct=True, use_hyponyms=True)
    ]
    golds = [p.larger for p in test_pairs]

    for setting in settings:
        preds = list()
        fractions_larger = list()
        notes = list()

        prop = VisualPropagation(G, config.visual_config)
        logger.info(f'\nRunning for setting {setting.print()}')
        comparer = Comparer(prop, setting, similar_words, objects)
        for test_pair in tqdm.tqdm(test_pairs):
            # TODO return confidence; use the higher one
            res_visual, fraction_larger, note = comparer.compare_visual_with_backoff(test_pair)
            fractions_larger.append(fraction_larger)
            preds.append(res_visual)
            notes.append(note)

        with open(f'visual_comparison_predictions_{setting.print()}.pkl', 'wb') as f:
            pickle.dump(list(zip(preds, fractions_larger, notes)), f)

        useful_counts = comparer.useful_paths_count
        tr = SymmetricalLogTransform(base=10, linthresh=1, linscale=1)
        ss = tr.transform([0., max(useful_counts) + 1])
        bins = tr.inverted().transform(np.linspace(*ss, num=100))
        fig, ax = plt.subplots()
        plt.hist(useful_counts, bins=bins)
        plt.xlabel('Number of useful paths')
        ax.set_xscale('symlog')
        plt.savefig(f'useful_paths{setting.print()}.png')

        useful_counts = np.array(useful_counts)
        logger.info(f'Number of objects with no useful path: {len(np.extract(useful_counts == 0, useful_counts))}')
        logger.info(f'Not recog count: {comparer.not_recognized_count}')

        logger.info(f'Total number of test cases: {len(golds)}')
        coverage, selectivity = coverage_accuracy_relational(golds, preds)
        logger.info(f'Coverage: {coverage}')
        logger.info(f'selectivity: {selectivity}')

        results.append(RelationalResult(setting.print(), selectivity, coverage))

        assert len(fractions_larger) == len(preds)
        corrects_not_none = list()
        diffs_not_none = list()
        for i, fraction_larger in enumerate(fractions_larger):
            gold = golds[i]
            res = preds[i]
            if fraction_larger is not None and fraction_larger != 0.5:
                fraction_larger_centered = fraction_larger - .5
                corrects_not_none.append(gold == res)
                diffs_not_none.append(abs(fraction_larger_centered))
        # TODO do something special for when fraction_larger_centered == 0

        regr_linear = Ridge(alpha=1.0)
        regr_linear.fit(np.reshape(diffs_not_none, (-1, 1)), corrects_not_none)
        with open('visual_confidence_model.pkl', 'wb') as f:
            pickle.dump(regr_linear, f)

        fig, ax = plt.subplots()
        bin_means, bin_edges, binnumber = stats.binned_statistic(diffs_not_none, corrects_not_none, 'mean',
                                                                 bins=20)
        bin_counts, _, _ = stats.binned_statistic(diffs_not_none, corrects_not_none, 'count',
                                                  bins=20)
        x = np.linspace(min(diffs_not_none), max(diffs_not_none), 500)
        X = np.reshape(x, (-1, 1))
        plt.plot(x, regr_linear.predict(X), '-', label='linear ridge regression')

        minc = min(bin_counts)
        maxc = max(bin_counts)
        norm = colors.SymLogNorm(vmin=minc, vmax=maxc, linthresh=1)
        bin_counts_normalized = [norm(c) for c in bin_counts]
        logger.info(f'counts, norm: {list(zip(bin_counts, bin_counts_normalized))}')
        viridis = cm.get_cmap('viridis', 20)

        mins = bin_edges[:-1]
        maxs = bin_edges[1:]
        mask = ~np.isnan(bin_means)
        plt.hlines(np.extract(mask, bin_means), np.extract(mask, mins), np.extract(mask, maxs),
                   colors=viridis(np.extract(mask, bin_counts_normalized)), lw=5,
                   label='binned statistic of data')
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
        ticks = [10**1.5, 10**1.75, 10**2, 10**2.5]
        colorbar = plt.colorbar(sm, ticks=ticks)
        colorbar.ax.set_yticklabels(['10^1.5', '10^1.75', '10^2', '10^2.5'])
        colorbar.set_label('bin count')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.xlabel('Absolute fraction_larger')
        plt.ylabel('Selectivity')
        ax.set_xscale('linear')
        plt.savefig('fraction_larger_selectivity_linear.png')
        plt.show()

        correlation, _ = pearsonr(diffs_not_none, corrects_not_none)
        logger.info(f'Pearsons correlation: {correlation}')

        correlation_spearman, _ = spearmanr(np.array(diffs_not_none), b=np.array(corrects_not_none))
        logger.info(f'Spearman correlation: {correlation_spearman}')

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_visual_backoff.csv')


class Comparer:
    def __init__(self, prop: VisualPropagation, setting: BackoffSettings, similar_words: dict, objects):

        self.objects = objects
        self.similar_words = similar_words
        self.setting = setting
        self.prop = prop
        self.useful_paths_count = []
        self.not_recognized_count = 0

    def compare_visual_with_backoff(self, test_pair) -> (bool, float):
        object1 = test_pair.e1.replace('_', ' ')
        object2 = test_pair.e2.replace('_', ' ')
        # TODO implement backoff mechanism
        recognizable_objects1, note1 = fill_objects_list(object1, self.setting, self.objects, self.similar_words)
        recognizable_objects2, note2 = fill_objects_list(object2, self.setting, self.objects, self.similar_words)
        comparisons = list()
        path_count_total = 0
        if len(recognizable_objects1) == 0 or len(recognizable_objects2) == 0:
            self.not_recognized_count += 1
        for o1 in recognizable_objects1:
            for o2 in recognizable_objects2:
                fraction_larger, path_count = self.prop.compare_pair(Pair(o1, o2))
                path_count_total += path_count
                if fraction_larger is not None:
                    comparisons.append(fraction_larger)
                logger.debug(f'{o1} {o2} fraction larger: {fraction_larger}')
        self.useful_paths_count.append(path_count_total)
        if len(comparisons) > 0:
            # larger_results = [c > .5 for c in comparisons]
            # fraction_larger_mean = np.mean(larger_results)
            fraction_larger_mean = np.mean(comparisons)
            res = fraction_larger_mean > .5
        else:
            res = None
            fraction_larger_mean = None
        note = f'Object 1: {note1} --- Object 2: {note2}'
        return res, fraction_larger_mean, note


def check_if_in_vg(word_list, vg_objects):
    res = list()
    for word in word_list:
        word = word.replace(' ', "_")
        if word in vg_objects:
            res.append(word)
    return res


def fill_objects_list(entity: str, setting: BackoffSettings, vg_objects: list, similar_words_lookup):
    synset_string = entity.replace(' ', '_')
    entity_string = entity.replace('_', " ")
    recognizable_objects = list()
    similar_words = similar_words_lookup[entity_string]
    # TODO check if this gets two-word objects!!
    note = ''
    if setting.use_direct:
        if synset_string in vg_objects:
            recognizable_objects.append(synset_string)
            note += f'used direct'

    if len(recognizable_objects) == 0 and setting.use_hyponyms:
        hyponyms = similar_words['hyponyms']
        if len(hyponyms) > 0:
            recognizable_objects += check_if_in_vg(hyponyms, vg_objects)
            note += f'used hyponyms: {hyponyms}'

    if len(recognizable_objects) == 0 and setting.use_hypernyms:
        hypernyms = similar_words['hypernyms']
        if len(hypernyms) > 0:
            recognizable_objects += check_if_in_vg(hypernyms, vg_objects)
            note += f'used hypernyms: {hypernyms}'

    if len(recognizable_objects) == 0 and setting.use_head_noun:
        head_nouns = similar_words['head_noun']
        assert type(head_nouns) is list
        if len(head_nouns) > 0:
            recognizable_objects += check_if_in_vg(head_nouns, vg_objects)
            note += f'used head nouns: {head_nouns}'

    if len(recognizable_objects) == 0 and setting.use_word2vec:
        word2vecs = similar_words['word2vec']
        if len(word2vecs) > 0:
            all_word2vecs_in_vg = check_if_in_vg(word2vecs, vg_objects)
            if len(all_word2vecs_in_vg) > 0:
                best_three = all_word2vecs_in_vg[:min(3, len(all_word2vecs_in_vg))]
                recognizable_objects += best_three
                note += f'used word2vecs: {best_three}'

    else:
        logger.debug(f'{synset_string} and fallback objects not in VG.')
    return recognizable_objects, note


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
