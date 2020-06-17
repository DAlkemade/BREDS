import logging
import pickle
import time
from textwrap import wrap
from typing import List

import tqdm
from spacy.tokens.doc import Doc

logger = logging.getLogger(__name__)

SAVE_STEP = 100
LIMIT = 500000


def parse_coref(htmls, nlp, name):
    name_coref_htmls = []
    for html in htmls:
        logger.info(f'html length: {len(html)}')
        try:
            if len(html) > LIMIT:
                htmls_split = wrap(html, width=LIMIT)
            else:
                htmls_split = [html]
            for h in htmls_split:
                html_coref = parse_doc(h, name, nlp)
                name_coref_htmls.append(html_coref)
        except MemoryError:
            pass

    return name_coref_htmls


def parse_doc(html, name, nlp):
    doc: Doc = nlp(html)
    clusters = doc._.coref_clusters
    relevant_clusters = []
    logger.debug(f'Resolve clusters for {name} of type {type(name)}')
    for cluster in clusters:
        text = cluster.main.text.lower()
        if name in text:
            relevant_clusters.append(cluster)
    return get_resolved(doc, relevant_clusters)


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


def find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp):
    timestamp = time.time()
    logger.info(f'Started at time {timestamp}')
    count_not_in_cache = 0
    for name in names:
        if name not in htmls_lookup_coref.keys():
            count_not_in_cache += 1

    logger.info(f'Doing coref; {count_not_in_cache} names are not in coref cache')

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
