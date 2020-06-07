import fileinput
import logging
import os
from collections import namedtuple

import pandas as pd
from logging_setup_dla.logging import set_up_root_logger

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
    for i in range(row_count):
        anouk_row: pd.Series = anouk.iloc[i]
        bram_row: pd.Series = bram.iloc[i]
        if anouk_row['object'] in remove:
            continue
        minimum = min(anouk_row['min'], bram_row['min'])
        maximum = max(anouk_row['max'], bram_row['max'])
        results.append(Result(anouk_row['object'], minimum, maximum))

    results_df = pd.DataFrame(results)
    results_df.set_index(['object'], inplace=True, drop=True)
    results_df.to_csv(os.path.join('data_numeric', 'VG_YOLO_intersection_test_annotated_combined.csv'))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
