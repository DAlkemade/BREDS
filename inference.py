import fileinput
import logging
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from functools import partial

from logging_setup_dla.logging import set_up_root_logger
from nltk.corpus import wordnet as wn

from breds.breds import process_objects, update_tuples_confidences
from breds.config import Weights, Config
from breds.htmls import scrape_htmls
from breds.similarity import similarity_all
from breds.visual import VisualConfig

set_up_root_logger(f'INFERENCE_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.getcwd())

logger = logging.getLogger(__name__)


def read_weights(parameters_fname: str):
    weights = Weights()
    for line in fileinput.input(parameters_fname):
        if line.startswith("alpha"):
            weights.alpha = float(line.split("=")[1])

        if line.startswith("beta"):
            weights.beta = float(line.split("=")[1])

        if line.startswith("gamma"):
            weights.gamma = float(line.split("=")[1])
    return weights


def main():
    cache_fname = 'inference_cache.pkl'
    if os.path.exists(cache_fname):
        with open(cache_fname, 'rb') as f:
            all_sizes = pickle.load(f)
    else:
        parser = ArgumentParser()
        parser.add_argument('--patterns', type=str, required=True)
        parser.add_argument('--unseen_objects', type=str, required=True)
        parser.add_argument('--configuration', type=str, required=True)
        parser.add_argument('--seeds_file', type=str, required=True)
        parser.add_argument('--negative_seeds', type=str, required=True)
        parser.add_argument('--similarity', type=float, required=True)
        parser.add_argument('--confidence', type=float, required=True)
        parser.add_argument('--objects', type=str, required=True)
        parser.add_argument('--cache_config_fname', type=str, required=True)
        parser.add_argument('--vg_objects', type=str, required=True)
        parser.add_argument('--vg_objects_anchors', type=str, required=True)
        args = parser.parse_args()
        patterns_fname = args.patterns
        unseen_objects_fname = args.unseen_objects
        with open(patterns_fname, 'rb') as f:
            patterns = pickle.load(f)
        # TODO check whether the objects aren't in the bootstrapped objects
        visual_config = VisualConfig(args.vg_objects, args.vg_objects_anchors)
        config = Config(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence,
                        args.objects, visual_config)
        unseen_objects = set([line.strip() for line in fileinput.input(unseen_objects_fname)])

        config.read_word2vec()

        similar_words = defaultdict(lambda: defaultdict(list))

        for entity in unseen_objects:
            # Word2vec
            # TODO the results for cheetah and container ship are not great, should probably be a last resort
            # TODO can be sped up if necessary:
            #  https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html#sphx-glr-auto-examples-tutorials-run-annoy-py
            most_similar = config.word2vec.most_similar(positive=entity.split(),
                                                        topn=5)  # TODO maybe use a bigram model? Because now those can not be entered and not be given as similar words
            most_similar = [m for m in most_similar if m[1] > .6]
            # logger.info(most_similar)
            words, _ = zip(*most_similar)
            similar_words[entity]['word2vec'] = words

            # TODO maybe use synset1.path_similarity(synset2)
            # Wordnet children / parents
            synsets = wn.synsets(entity.replace(' ', '_'), pos=wn.NOUN)
            # TODO think about homonyms
            for synset in synsets:
                hypernyms = [s.lemma_names()[0].replace('_', ' ') for s in synset.hypernyms()]  # only use one name
                hyponyms = [s.lemma_names()[0].replace('_', ' ') for s in synset.hyponyms()]
                # TODO set a limit on the number of hyponyms, e.g. 'animal' might have thousands
                # logger.info(synset.lexname())
                similar_words[entity]['hyponyms'] += hyponyms
                similar_words[entity]['hypernyms'] += hypernyms

        logger.info(similar_words)

        # Create object lookup
        objects_lookup = defaultdict(list)
        for main_object, related_words_dict in similar_words.items():
            for type, values in related_words_dict.items():
                for related_word in values:
                    objects_lookup[related_word].append(main_object)
        logger.info(objects_lookup)

        all_new_objects = set(objects_lookup.keys()).union(unseen_objects)

        # TODO use similar_words

        htmls_lookup = scrape_htmls('htmls_unseen_objects.pkl', list(all_new_objects))

        # TODO coreferences
        logger.info(f'Using coreference: {config.coreference}')
        tuples = process_objects(all_new_objects, htmls_lookup, config)

        candidate_tuples = defaultdict(list)
        for t in tuples:
            for extraction_pattern in patterns:
                accept, score = similarity_all(
                    t, extraction_pattern, config.weights, config.threshold_similarity
                )
                if accept:
                    candidate_tuples[t].append(
                        (extraction_pattern, score)
                    )

        update_tuples_confidences(candidate_tuples, config)

        # Print tuples
        extracted_tuples = list(candidate_tuples.keys())
        tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence,
                               reverse=True)

        # Compile all results for each object
        all_sizes = defaultdict(partial(defaultdict, list))
        for t in tuples_sorted:
            entity = t.e1
            relevant_objects = objects_lookup[entity]
            for relevant_object in relevant_objects:
                og_dict = similar_words[relevant_object]
                for type, values in og_dict.items():
                    if entity in values:
                        all_sizes[relevant_object][type].append(t)

            if entity in unseen_objects:
                all_sizes[entity]['itself'].append(t)
        with open(cache_fname, 'wb') as f:
            pickle.dump(all_sizes, f, pickle.HIGHEST_PROTOCOL)

    logger.info(all_sizes)
    for object, sims_dict in all_sizes.items():
        logger.info(f'Processing for {object}')
        for key, values in sims_dict.items():
            logger.info(f'\n{key} finds:')
            for t in values:
                logger.info(t.sentence)
                logger.info(f"{t.e1} {t.e2} with confidence {t.confidence}")

        # TODO use results in an order, e.g direct finds -> mean of hyponyms -> mean of hypernyms -> word2vec
        #  Maybe this is something I should experiment with

    logger.info('Finished')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
