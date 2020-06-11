import fileinput
import logging
import os
from collections import namedtuple
from math import ceil, floor

import pandas as pd
from logging_setup_dla.logging import set_up_root_logger
from matplotlib.scale import SymmetricalLogTransform
from matplotlib import pyplot as plt
import numpy as np

set_up_root_logger('ANALYZETEST', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


def main():
    anouk: pd.DataFrame = pd.read_csv('data_numeric/VG_YOLO_intersection_test_annotated_anouk.csv')
    anouk: pd.DataFrame = anouk.astype({'object': str})
    anouk.set_index(['object'], inplace=True, drop=False)
    bram: pd.DataFrame = pd.read_csv('data_numeric/VG_YOLO_intersection_test_annotated_bram.csv')
    bram: pd.DataFrame = bram.astype({'object': str})
    bram.set_index(['object'], inplace=True, drop=False)
    assert len(anouk.keys()) == len(bram.keys())

    bram_no_size = [line.strip().lower() for line in fileinput.input('data_numeric/hard_words_bram.txt')]
    anouk_no_size = [line.strip().lower() for line in fileinput.input('data_numeric/hard_words_anouk.txt')]
    remove = ['snow', 'architecture','toilet water']

    Result = namedtuple('Result', ['object', 'min', 'max'])
    results = list()
    row_count = len(anouk.index)
    min_diffs = list()
    max_diffs = list()
    maxmin = ('None', 0)
    maxmax = ('None', 0)
    fraction_overlaps = list()
    for i in range(row_count):
        anouk_row: pd.Series = anouk.iloc[i]
        bram_row: pd.Series = bram.iloc[i]
        if anouk_row['object'] in remove:
            continue
        anouk_min_size: int = anouk_row['min']
        anouk_max_size: int = anouk_row['max']
        bram_min_size: int = bram_row['min']
        bram_max_size: int = bram_row['max']

        minimum = min(anouk_min_size, bram_min_size)
        maximum = max(anouk_max_size, bram_max_size)
        results.append(Result(anouk_row['object'], minimum, maximum))
        mindif =anouk_min_size-bram_min_size
        maxdif = anouk_max_size - bram_max_size
        if abs(mindif) > maxmin[1]:
            maxmin = (anouk_row['object'], abs(mindif))
        if abs(maxdif) > maxmax[1]:
            maxmax = (anouk_row['object'], abs(maxdif))
        min_diffs.append(mindif)
        max_diffs.append(maxdif)

        agreement_min = max(anouk_min_size, bram_min_size)
        agreement_max = min(anouk_max_size, bram_max_size)
        if agreement_min > agreement_max:
            fraction_overlap = 0.
        else:
            fraction_overlap = (agreement_max-agreement_min)/(maximum-minimum)
        fraction_overlaps.append(fraction_overlap)


    results_df = pd.DataFrame(results)
    results_df.set_index(['object'], inplace=True, drop=True)
    results_df.to_csv(os.path.join('data_numeric', 'VG_YOLO_intersection_test_annotated_combined.csv'))

    create_hist(min_diffs, max_diffs)

    logger.info(f'maximum of min differences: {maxmin}')
    logger.info(f'maximum of max differences: {maxmax}')

    fig, ax = plt.subplots()
    plt.hist(fraction_overlaps, bins=10)
    plt.xlabel('Jaccard index')
    plt.ylabel('count')
    plt.show()




def create_hist(mins: list, maxs: list):
    mins = np.absolute(mins)
    maxs = np.absolute(maxs)
    comp = []
    comp.append(mins)
    comp.append(maxs)
    min_value = min(mins + maxs)
    max_value =  max(mins + maxs)
    lintresh = .01
    if min_value <= 0:
        tr = SymmetricalLogTransform(base=10, linthresh=lintresh, linscale=1)
        ss = tr.transform([min_value-1, max_value + 1])
        bins = tr.inverted().transform(np.linspace(*ss, num=15))
    else:
        bins = np.logspace(floor(np.log10(min_value)), ceil(np.log10(max_value)), 15, base=10)
    plt.style.use('seaborn-deep')

    fig, ax = plt.subplots()
    plt.hist(comp, bins, label=['minimum sizes', 'maximum sizes'])
    # plt.axvline(np.mean(mins), color='b', linestyle='dashed', linewidth=1)
    # plt.axvline(np.mean(maxs), color='g', linestyle='dashed', linewidth=1)

    plt.legend(loc='upper right')
    ax.set_xscale('symlog', linthreshx=lintresh)
    plt.xlim(left=0.)
    plt.xlabel('Absolute difference [m]')
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
