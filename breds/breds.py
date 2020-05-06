import json
import logging
import operator
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import List

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from nltk import load, tokenize

from breds.config import Config
from breds.htmls import scrape_htmls
from breds.pattern import Pattern, pattern_factory
from breds.sentence import Sentence
from breds.similarity import similarity_all
from breds.tuple import Tuple
from breds.visual import VisualConfig, check_tuple_with_visuals

logger = logging.getLogger(__name__)
PRINT_TUPLES = True
PRINT_PATTERNS = True
PRINT_SEED_MATCHES = False


def print_tuple_props(t: Tuple):
    logger.info(f'tuple: {t.e1} {t.e2}')
    logger.info(f'before | vector: {t.bef_vector} | words: {t.bef_words} | tags: {t.bef_tags}')
    logger.info(f'between | vector: {t.bet_vector} | words: {t.bet_words} |  tags: {t.bet_tags}')
    logger.info(f'after | vector: {t.aft_vector} | words: {t.aft_words} | tags: {t.aft_tags}')


def update_tuples_confidences(candidate_tuples: dict, config: Config):
    for t in list(candidate_tuples.keys()):
        confidence = 1
        t.confidence_old = t.confidence
        for p in candidate_tuples.get(t):
            confidence *= 1 - (p[0].confidence * p[1])
        t.confidence = 1 - confidence
        # TODO maybe only do this for candates which otherwise would have been added to seed
        # TODO think about the following: this sort of promotes tuples that are not recognized by the detector, because they won't be removed
        if config.visual:

            corrects = check_tuple_with_visuals(config.visual_config, t.e1, t.e2)
            total_hits = len(corrects)

            if total_hits > 3:  # only use visual if enough comparisons were used
                visual_confidence = np.mean(corrects)
                logger.info(
                    f'Used visual for tuple: {total_hits} hits; visual confidence {visual_confidence}; normal confidence: {t.confidence}')
                if visual_confidence < config.visual_cutoff:  # TODO think about visual cutoff
                    if t.confidence > config.instance_confidence:
                        logger.info(f'Prevented tuple {t.e1} {t.e2} from being added to seeds with visuals')
                    t.confidence = 0.


class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidence, objects, vg_objects, vg_objects_anchors):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        visual_config = VisualConfig(vg_objects, vg_objects_anchors)
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidence, objects, visual_config)
        # TODO change to full matrix

    def generate_tuples(self, htmls_fname: str, tuples_fname: str):
        """
        Generate tuples instances from a text file with sentences where named entities are
        already tagged

        :param sentences_file:
        """

        if os.path.exists(tuples_fname):
            with open(tuples_fname, "rb") as f_in:
                logger.info("\nLoading processed tuples from disk...")
                self.processed_tuples = pickle.load(f_in)
            logger.info(f"{len(self.processed_tuples)} tuples loaded")

        else:

            # load needed stuff, word2vec model and a pos-tagger

            logger.info("\nGenerating relationship instances from sentences")
            names = list(self.config.objects)

            if os.path.exists(htmls_fname):
                logger.info("Loading htmls from disk")
                with open(htmls_fname, "rb") as f_html:
                    htmls_lookup = pickle.load(f_html)
            else:
                logger.info("Retrieving htmls")
                if self.config.coreference:
                    raise ValueError(
                        'We have not implemented lazy coreferences. You need to parse these in advance on a GPU.')
                htmls_lookup = scrape_htmls(htmls_fname, names)

            logger.info(f'Using coreference: {self.config.coreference}')
            self.config.read_word2vec()

            self.processed_tuples += process_objects(names, htmls_lookup, self.config)

            logger.info(f"\n{len(self.processed_tuples)} tuples generated")

            print("Writing generated tuples to disk")
            with open(tuples_fname, "wb") as f_out:
                pickle.dump(self.processed_tuples, f_out)

        object_occurrence = dict()
        for o in self.config.objects:
            object_occurrence[o] = 0
        for t in self.processed_tuples:
            try:
                object_occurrence[t.e1] += 1
            except KeyError:
                logger.warning(f'{t.e1} not in objects')
        occurrences = list(object_occurrence.values())
        max_value = 100
        bins = np.linspace(0, max_value, max_value+1)
        clipped_values = np.clip(occurrences, bins[0], bins[-1])
        hist, _, _ = plt.hist(clipped_values, bins=bins)
        logger.info(f'Number of objects with no tuple: {hist[0]}')
        plt.show()


    def match_seeds_tuples(self):
        # checks if an extracted tuple matches seeds tuples
        if self.config.e1_type != 'OBJECT' or self.config.e2_type != 'NUMBER':
            raise RuntimeError("This function is only suitable for object-numer combinations")
        matched_tuples = list()
        count_matches = dict()
        for t in self.processed_tuples:
            for s in self.config.positive_seed_tuples.values():
                # only match the first, as long is the second is a number
                try:
                    float(t.e2)
                except ValueError:
                    # TODO catch correct error
                    logging.WARN(f'Couldnt cast string to float: {t.e2}')
                    continue
                if t.e1 == s.e1:
                    matched_tuples.append(t)
                    try:
                        count_matches[(t.e1, t.e2)] += 1
                    except KeyError:
                        count_matches[(t.e1, t.e2)] = 1

        return count_matches, matched_tuples

    def write_relationships_to_disk(self):
        logger.info("\nWriting extracted relationships to disk")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        f_output = open(os.path.join('relationships_output', f"relationships{timestr}.txt"), "w")
        tmp = sorted(list(self.candidate_tuples.keys()), reverse=True)
        for t in tmp:
            f_output.write("instance: " + t.e1 + '\t' + str(t.e2) + '\tscore:' + str(t.confidence) + '\n')
            try:
                # f_output.write("sentence: "+t.sentence+'\n')
                f_output.write("pattern_bef: " + t.bef_words + '\n')
                f_output.write("pattern_bet: " + t.bet_words + '\n')
                f_output.write("pattern_aft: " + t.aft_words + '\n')
            except UnicodeEncodeError:
                f_output.write("cant encode one of the words in unicode" + '\n')

            if t.passive_voice is False:
                f_output.write("passive voice: False\n")
            elif t.passive_voice is True:
                f_output.write("passive voice: True\n")
            f_output.write("\n")
        f_output.close()

    def init_bootstrap(self, tuples):

        # starts a bootstrap iteration

        if tuples is not None:
            f = open(tuples, "r")
            logger.info("\nLoading processed tuples from disk...")
            self.processed_tuples = pickle.load(f)
            f.close()
        logger.info(f"{len(self.processed_tuples)} tuples loaded")



        self.curr_iteration = 0
        try:
            while self.curr_iteration <= self.config.number_iterations:
                logger.info("==========================================")
                logger.info(f"\nStarting iteration {self.curr_iteration}")
                logger.info(f"\nLooking for seed matches of:")
                for s in self.config.positive_seed_tuples.values():
                    logger.info(f"{s.e1} \t {s.sizes}")

                # Looks for sentences matching the seed instances
                count_matches, matched_tuples = self.match_seeds_tuples()

                if len(matched_tuples) == 0:
                    logger.info("\nNo seed matches found")
                    sys.exit(0)

                else:
                    logger.info("\nNumber of seed matches found")
                    sorted_counts = sorted(
                        list(count_matches.items()),
                        key=operator.itemgetter(1),
                        reverse=True
                    )
                    if PRINT_SEED_MATCHES:
                        for t in sorted_counts:
                            logger.info(f"{t[0][0]} \t {t[0][1]} {t[1]}")

                    matched_seed_objects: set = set()
                    for t in sorted_counts:
                        entity = t[0][0]
                        matched_seed_objects.add(entity)
                    logger.info(f'{len(matched_seed_objects)} unique seed objects matched')
                    logger.info(f'Matched seed objects: {matched_seed_objects}')
                    logger.info(f"\n {len(matched_tuples)} tuples matched")

                    # Cluster the matched instances, to generate
                    # patterns/update patterns
                    logger.info(
                        f"\nClustering matched instances to generate patterns in iteration {self.curr_iteration}")
                    self.cluster_tuples(matched_tuples)

                    # Eliminate patterns supported by less than
                    # 'min_pattern_support' tuples
                    new_patterns = [p for p in self.patterns if len(p.tuples) >
                                    self.config.min_pattern_support]
                    self.patterns: List[Pattern] = new_patterns

                    logger.info(f"\n{len(self.patterns)} patterns generated")

                    # if PRINT_PATTERNS is True:
                    #     count = 1
                    #     logger.info("\nPatterns:")
                    #     for p in self.patterns:
                    #         logger.info(count)
                    #         for t in p.tuples:
                    #             logger.info(f"e1 {t.e1}")
                    #             logger.info(f"e2 {t.e2}")
                    #             logger.info(f"BEF {t.bef_words}")
                    #             logger.info(f"BET {t.bet_words}")
                    #             logger.info(f"AFT {t.aft_words}")
                    #             logger.info("========")
                    #             logger.info("\n")
                    #         count += 1

                    if self.curr_iteration == 0 and len(self.patterns) == 0:
                        logger.info("No patterns generated")
                        sys.exit(0)

                    # Look for sentences with occurrence of seeds
                    # semantic types (e.g., ORG - LOC)
                    # This was already collect and its stored in:
                    # self.processed_tuples
                    #
                    # Measure the similarity of each occurrence with each
                    # extraction pattern and store each pattern that has a
                    # similarity higher than a given threshold
                    #
                    # Each candidate tuple will then have a number of patterns
                    # that extracted it each with an associated degree of match.
                    logger.info(f"Number of tuples to be analyzed: {len(self.processed_tuples)}")

                    logger.info(
                        f"\nCollecting instances based on extraction patterns in iteration {self.curr_iteration}")

                    for t in tqdm.tqdm(self.processed_tuples):

                        sim_best = 0
                        for extraction_pattern in self.patterns:
                            accept, score = similarity_all(
                                t, extraction_pattern, self.config.weights, self.config.threshold_similarity
                            )
                            if accept is True:
                                extraction_pattern.update_selectivity(
                                    t, self.config
                                )
                                if score > sim_best:
                                    sim_best = score
                                    pattern_best = extraction_pattern

                        if sim_best >= self.config.threshold_similarity:
                            # if this tuple was already extracted, check if this
                            # extraction pattern is already associated with it,
                            # if not, associate this pattern with it and store the
                            # similarity score
                            patterns = self.candidate_tuples[t]
                            if patterns is not None:
                                if pattern_best not in [x[0] for x in patterns]:
                                    self.candidate_tuples[t].append(
                                        (pattern_best, sim_best)
                                    )

                            # If this tuple was not extracted before
                            # associate this pattern with the instance
                            # and the similarity score
                            else:
                                self.candidate_tuples[t].append(
                                    (pattern_best, sim_best)
                                )

                    # update all patterns confidence
                    for p in self.patterns:
                        p.update_confidence(self.config)

                    if PRINT_PATTERNS is True:
                        logger.info("\nPatterns:")
                        for p in self.patterns:
                            for t in p.tuples:
                                logger.info(f"BEF {t.bef_words}")
                                logger.info(f"BET {t.bet_words}")
                                logger.info(f"AFT {t.aft_words}")
                                logger.info("========")
                            logger.info(f"Tuples {len(p.tuples)}")
                            logger.info(f"Pattern Confidence {p.confidence}")
                            logger.info(f"\n")

                    # update tuple confidence based on patterns confidence
                    logger.info("\n\nCalculating tuples confidence")
                    update_tuples_confidences(self.candidate_tuples, self.config)

                    logger.info(f'Number of entities with at least one match: {len(self.candidate_tuples.keys())}')
                    # sort tuples by confidence and print
                    if PRINT_TUPLES is True:
                        extracted_tuples = list(self.candidate_tuples.keys())
                        tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence,
                                               reverse=True)
                        for t in tuples_sorted:
                            logger.info(t.sentence)
                            logger.info(f"{t.e1} {t.e2}")
                            logger.info(t.confidence)
                            logger.info("\n")

                    logger.info("Adding tuples to seed with confidence >= {}".format(
                        str(self.config.instance_confidence)))
                    for t in list(self.candidate_tuples.keys()):
                        if t.confidence >= self.config.instance_confidence:
                            self.config.add_seed_to_dict(t.e1, t.e2, self.config.positive_seed_tuples)

                    # increment the number of iterations
                    self.curr_iteration += 1
        except KeyboardInterrupt:
            pass

        self.write_relationships_to_disk()
        self.write_seeds_to_disk()
        self.write_patterns_to_disk()

    def cluster_tuples(self, matched_tuples):
        # this is a single-pass clustering
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            logger.info("There are no patterns, so creating one")
            c1 = pattern_factory(self.config, matched_tuples[0])
            self.patterns.append(c1)

        for t in tqdm.tqdm(matched_tuples):
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one
            # with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = similarity_all(t, extraction_pattern, self.config.weights, self.config.threshold_similarity)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having
            #  this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = pattern_factory(self.config, t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with
            # the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

    def write_seeds_to_disk(self):
        logger.info('Saving seeds to disk')
        timestr = time.strftime("%Y%m%d-%H%M%S")
        printable_seed_dict = dict((k, list(v.sizes)) for k,v in self.config.positive_seed_tuples.items())
        with open(f'final_seeds_{timestr}.json', 'w') as outfile:
            json.dump(printable_seed_dict, outfile)

    def write_patterns_to_disk(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open(f'patterns_{timestr}.pkl', 'wb') as f:
            pickle.dump(self.patterns, f, pickle.HIGHEST_PROTOCOL)


def process_objects(names: list, htmls_lookup: dict, config: Config):
    tuples = list()
    logger.info("Start parsing tuples")
    tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')

    for object in tqdm.tqdm(names):
        # TODO think about units. could something automatic be done? it should in theory be possible to learn the meaning of each unit
        # otherwise reuse the scraper pattern to only find numbers with a length unit for now
        # TODO I might have to do recognition of 'they' etc. e.g. for lion: With a typical head-to-body length of 184–208 cm (72–82 in) they are larger than females at 160–184 cm (63–72 in).
        # or 'Generally, males vary in total length from 250 to 390 cm (8.2 to 12.8 ft)'  for tiger
        # TODO think about plurals, e.g. tigers
        try:
            htmls: List[str] = htmls_lookup[object]
        except KeyError:
            logger.warning(f'No htmls for {object}')
            continue

        for html in htmls:
            sentences = tokenize.sent_tokenize(html)

            # TODO split sentences from docs
            for line in sentences:
                line = line.lower()  # TODO should I do this?

                # TODO here I should change how tuples are found (i.e. all combinations of anchor objects)
                sentence = Sentence(line.strip(),
                                    config.e1_type,
                                    config.e2_type,
                                    config.max_tokens_away,
                                    config.min_tokens_away,
                                    config.context_window_size, object, tagger,
                                    config)

                for rel in sentence.relationships:
                    t = Tuple(rel.e1, rel.e2,
                              rel.sentence, rel.before, rel.between, rel.after,
                              config)
                    tuples.append(t)
    return tuples