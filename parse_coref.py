import logging
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import neuralcoref
import spacy

from breds.config import read_objects_of_interest, parse_objects_from_seed
from logging_setup_dla.logging import set_up_root_logger

from breds.coref import find_corefs

logger = logging.getLogger(__name__)


def main():
    logger.info("Start coreference parsing")
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    parser.add_argument('--htmls_coref_cache', type=str, required=True)
    parser.add_argument('--work_dir', type=str, required=False, default=os.getcwd())
    args = parser.parse_args()
    work_dir = args.work_dir
    set_up_root_logger('COREF', os.path.join(work_dir, 'logs'))

    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)
    htmls_coref_cache_fname: str = args.htmls_coref_cache

    with open(html_fname, "rb") as f_html:
        htmls_lookup = pickle.load(f_html)

    htmls_lookup_coref = load_cache(htmls_coref_cache_fname)

    names = get_all_objects(objects_path, work_dir)
    logger.info(f'Number of objects: {len(names)}')

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)

    find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp)

    with open(htmls_coref_cache_fname, 'wb') as f:
        pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Finished')


def get_all_objects(objects_path, work_dir: str = '') -> list:
    seeds: set = parse_objects_from_seed(os.path.join(work_dir, 'data_numeric/seeds_positive.txt')).union(parse_objects_from_seed(
        os.path.join(work_dir, 'data_numeric/seeds_negative.txt')))
    names = list(read_objects_of_interest(objects_path).union(seeds))
    return names


def load_cache(htmls_coref_cache_fname):
    if os.path.exists(htmls_coref_cache_fname):
        with open(htmls_coref_cache_fname, "rb") as f:
            htmls_lookup_coref = pickle.load(f)
    else:
        htmls_lookup_coref = dict()
    return htmls_lookup_coref


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
