import fileinput
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import neuralcoref
import spacy
import sys
from typing import List

import tqdm



def parse_coref(htmls_lookup, names):
    coreference_times: List[float] = list()

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    htmls_coref = dict()
    for name in tqdm.tqdm(names):
        try:
            htmls: List[str] = htmls_lookup[name]
        except KeyError:
            print(f'No htmls for {name}')
            continue
        name_coref_htmls = []
        for html in htmls:
            time_before = time.time()
            print(f'html length: {len(html)}')
            try:
                doc = nlp(html)
                html = doc._.coref_resolved
                coreference_times.append(time.time() - time_before)
                name_coref_htmls.append(html)
            except MemoryError:
                pass
        htmls_coref[name] = name_coref_htmls
    coref_comp = np.mean(coreference_times)
    print(f'Average coreference comp time: {coref_comp}')
    f = open('coref_computation_time.txt', 'w')
    f.write(f'coref_comp: {coref_comp}')
    f.close()
    return htmls_coref

def main():
    parser = ArgumentParser()
    parser.add_argument('--htmls_fname', type=str, required=True)
    parser.add_argument('--objects_fname', type=str, required=True)
    args = parser.parse_args()
    html_fname: str = args.htmls_fname
    objects_path = Path(args.objects_fname)

    with open(html_fname, "rb") as f_html:
        htmls_lookup = pickle.load(f_html)

    objects = set()
    for line in fileinput.input(objects_path):
        object = line.strip().lower()
        objects.add(object)
    names = list(objects)


    htmls_lookup_coref = parse_coref(htmls_lookup, names)

    with open(f'{html_fname.split(".")[0]}_coref.pkl', 'wb') as f:
        pickle.dump(htmls_lookup_coref, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()