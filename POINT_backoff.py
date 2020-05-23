import json
import logging
import os
import pickle
from datetime import datetime

import pandas as pd
import yaml
from box import Box
from learning_sizes_evaluation.evaluate import precision_recall, range_distance
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame

from breds.breds_inference import predict_sizes, gather_sizes_with_bootstrapping_patterns, compile_results, \
    find_similar_words, create_reverse_lookup, load_patterns
from breds.config import load_word2vec

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
    if "data_numeric/VG_YOLO_intersection_dev_annotated.csv" in input_fname and os.path.exists(cache_fname):
        with open(cache_fname, 'rb') as f:
            all_sizes = pickle.load(f)


    else:
        word2vec_model = load_word2vec(cfg.parameters.word2vec_path)
        similar_words = find_similar_words(word2vec_model, unseen_objects)

        # Create object lookup
        objects_lookup = create_reverse_lookup(similar_words)

        all_new_objects = set(objects_lookup.keys()).union(unseen_objects)

        tuples_bootstrap = gather_sizes_with_bootstrapping_patterns(cfg, patterns, all_new_objects)

        all_sizes = compile_results(tuples_bootstrap, objects_lookup, similar_words, unseen_objects)
        with open(cache_fname, 'wb') as f:
            pickle.dump(all_sizes, f)

    predictions = predict_sizes(all_sizes, unseen_objects)

    with open(f'backoff_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    precision_recall(input, predictions)
    range_distance(input, predictions)

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
