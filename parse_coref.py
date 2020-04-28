import logging
import os
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import neuralcoref
import spacy
import tqdm
from spacy.tokens import Doc

from breds.config import read_objects_of_interest, parse_objects_from_seed
from logging_setup import set_up_logging

logger = logging.getLogger(__name__)

SAVE_STEP = 100


def parse_coref(htmls, nlp, name):
    name_coref_htmls = []
    for html in htmls:
        logger.info(f'html length: {len(html)}')
        try:
            doc: Doc = nlp(html)
            clusters = doc._.coref_clusters
            relevant_clusters = []
            for cluster in clusters:
                text = cluster.main.text.lower()
                if name in text:
                    relevant_clusters.append(cluster)
            html_coref = get_resolved(doc, relevant_clusters)
            name_coref_htmls.append(html_coref)
        except MemoryError:
            pass

    return name_coref_htmls


def get_resolved(doc, clusters):
    ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
    resolved = list(tok.text_with_ws for tok in doc)
    for cluster in clusters:
        for coref in cluster:
            if coref != cluster.main:
                resolved[coref.start] = cluster.main.text + doc[coref.end - 1].whitespace_
                for i in range(coref.start + 1, coref.end):
                    resolved[i] = ""
    return ''.join(resolved)


def main():
    logger.info("Start coreference parsing")
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    parser.add_argument('--htmls_coref_cache', type=str, required=True)
    parser.add_argument('--work_dir', type=str, required=True)
    args = parser.parse_args()
    set_up_logging('COREF', os.path.join(args.work_dir, 'logs'))

    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)
    htmls_coref_cache_fname: str = args.htmls_coref_cache

    with open(html_fname, "rb") as f_html:
        htmls_lookup = pickle.load(f_html)

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)

    htmls_lookup_coref = load_cache(htmls_coref_cache_fname)

    names = get_all_objects(objects_path)
    logger.info(f'Number of objects: {len(names)}')

    find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp)

    with open(htmls_coref_cache_fname, 'wb') as f:
        pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Finished')


def get_all_objects(objects_path) -> list:
    seeds: set = parse_objects_from_seed('data_numeric/seeds_positive.txt').union(parse_objects_from_seed(
        'data_numeric/seeds_negative.txt'))
    names = list(read_objects_of_interest(objects_path).union(seeds))
    return names


def find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp):
    timestamp = time.time()
    logger.info(f'Started at time {timestamp}')
    for name in tqdm.tqdm(names):
        if name in htmls_lookup_coref.keys():
            continue
        try:
            htmls: List[str] = htmls_lookup[name]
        except KeyError:
            logger.warning(f'No htmls for {name}')
            continue
        htmls_coref = parse_coref(htmls, nlp, name)
        htmls_lookup_coref[name] = htmls_coref
        if time.time() - timestamp > 50 * 60:
            logger.info("Saving intermediate results")
            with open(htmls_coref_cache_fname, 'wb') as f:
                pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)
            timestamp = time.time()


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
