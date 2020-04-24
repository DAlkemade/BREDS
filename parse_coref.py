import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import neuralcoref
import spacy
import tqdm

from breds.config import read_objects_of_interest

SAVE_STEP = 100

def parse_coref(htmls, nlp):
    name_coref_htmls = []
    for html in htmls:
        print(f'html length: {len(html)}')
        try:
            doc = nlp(html)
            html_coref = doc._.coref_resolved
            name_coref_htmls.append(html_coref)
        except MemoryError:
            pass

    return name_coref_htmls


def main():
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    parser.add_argument('--htmls_coref_cache', type=str, required=True)
    args = parser.parse_args()
    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)
    htmls_coref_cache_fname: str = args.htmls_coref_cache

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)

    with open(html_fname, "rb") as f_html:
        htmls_lookup = pickle.load(f_html)

    htmls_lookup_coref = load_cache(htmls_coref_cache_fname)

    names = list(read_objects_of_interest(objects_path))
    print(f'Number of objects: {len(names)}')

    find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp)

    with open(htmls_coref_cache_fname, 'wb') as f:
        pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)

    print('Finished')


def find_corefs(htmls_coref_cache_fname, htmls_lookup, htmls_lookup_coref, names, nlp):
    count = 0
    for name in tqdm.tqdm(names):
        if name in htmls_lookup_coref.keys():
            continue
        try:
            htmls: List[str] = htmls_lookup[name]
        except KeyError:
            print(f'No htmls for {name}')
            continue
        htmls_coref = parse_coref(htmls, nlp)
        htmls_lookup_coref[name] = htmls_coref
        count += 1
        if count % SAVE_STEP == 0:
            with open(htmls_coref_cache_fname, 'wb') as f:
                pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)


def load_cache(htmls_coref_cache_fname):
    if os.path.exists(htmls_coref_cache_fname):
        with open(htmls_coref_cache_fname, "rb") as f:
            htmls_lookup_coref = pickle.load(f)
    else:
        htmls_lookup_coref = dict()
    return htmls_lookup_coref


if __name__ == "__main__":
    main()
