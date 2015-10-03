#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import cPickle
import sys
import codecs
import operator
import math
import multiprocessing
import Queue

from numpy import dot, asarray
from gensim import matutils
from collections import defaultdict
from nltk.data import load

from BREDS.Pattern import Pattern
from BREDS.Config import Config
from BREDS.Tuple import Tuple
from BREDS.Sentence import Sentence
from BREDS.Seed import Seed

# usefull stuff for debugging
PRINT_TUPLES = False
PRINT_PATTERNS = False


class BREDS(object):

    def __init__(self, config_file, seeds_file, negative_seeds, similarity, confidance, num_cores):
        if num_cores == 0:
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = num_cores
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.curr_iteration = 0
        self.patterns = list()
        self.patterns_index = dict()
        self.config = Config(config_file, seeds_file, negative_seeds, similarity, confidance)

    def generate_tuples(self, sentences_file):
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged
        """
        self.config.read_word2vec()     # load word2vec model

        # copy all sentences from input file into a Queue shared by all processes
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        print "\nLoading sentences from file"
        f_sentences = codecs.open(sentences_file, encoding='utf-8')
        count = 0
        for line in f_sentences:
            if line.startswith("#"):
                continue
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")
            queue.put(line.strip())
        f_sentences.close()

        pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
        processes = [multiprocessing.Process(target=self.generate_instances, args=(queue, pipes[i][1]))
                     for i in range(self.num_cpus)]

        print "\nGenerating relationship instances from sentences"
        print "Running", len(processes), " processes"
        for proc in processes:
            proc.start()

        for i in range(len(pipes)):
            data = pipes[i][0].recv()
            child_instances = data[1]
            for x in child_instances:
                self.processed_tuples.append(x)

        for proc in processes:
            proc.join()

        print "\n", len(self.processed_tuples), "instances generated"
        print "Writing generated tuples to disk"
        f = open("processed_tuples.pkl", "wb")
        cPickle.dump(self.processed_tuples, f)
        f.close()

    def generate_instances(self, sentences, child_conn):
        # Each process has its own NLTK PoS-tagger
        tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
        instances = list()
        while True:
            try:
                s = sentences.get_nowait()
                if sentences.qsize() % 500 == 0:
                    print multiprocessing.current_process(), "Instances to process", sentences.qsize()

                sentence = Sentence(s, self.config.e1_type, self.config.e2_type, self.config.max_tokens_away,
                                    self.config.min_tokens_away, self.config.context_window_size, tagger,
                                    self.config)

                for rel in sentence.relationships:
                    t = Tuple(rel.e1, rel.e2, rel.sentence, rel.before, rel.between, rel.after, self.config)
                    instances.append(t)

            except Queue.Empty:
                print multiprocessing.current_process(), "Queue is Empty"
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, instances))
                break

    def similarity_3_contexts_fast(self, p, t):
        (bef, bet, aft) = (0, 0, 0)

        # transform tuple into numpy array
        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(asarray(p.bef_vector)))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(asarray(p.bet_vector)))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(asarray(p.aft_vector)))

        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft

    def similarity_3_contexts(self, t, p):
        (bef, bet, aft) = (0, 0, 0)

        if t.bef_vector is not None and p.bef_vector is not None:
            bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

        if t.bet_vector is not None and p.bet_vector is not None:
            bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

        if t.aft_vector is not None and p.aft_vector is not None:
            aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

        return self.config.alpha*bef + self.config.beta*bet + self.config.gamma*aft

    def similarity_all(self, t, extraction_pattern):
        """
        Cosine similarity between all patterns part of a Cluster/Extraction Pattern
        and the vector of a ReVerb pattern extracted from a sentence, returns the max
        """
        good = 0
        bad = 0
        max_similarity = 0

        if self.curr_iteration == 0:
            for p in list(extraction_pattern.tuples):
                score = self.similarity_3_contexts(t, p)
                if score > max_similarity:
                    max_similarity = score
                if score >= self.config.threshold_similarity:
                    good += 1
                else:
                    bad += 1
        else:
            for p in list(extraction_pattern.tuples_vectors_uniques):
                score = self.similarity_3_contexts_fast(t, p)

                if score > max_similarity:
                    max_similarity = score
                if score >= self.config.threshold_similarity:
                    good += 1
                else:
                    bad += 1

        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0

    def match_seeds_tuples(self):
        """
        Checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = list()
        count_matches = dict()
        for t in self.processed_tuples:
            for s in self.config.positive_seed_tuples:
                if t.e1 == s.e1 and t.e2 == s.e2:
                    matched_tuples.append(t)
                    try:
                        count_matches[(t.e1, t.e2)] += 1
                    except KeyError:
                        count_matches[(t.e1, t.e2)] = 1
        return count_matches, matched_tuples

    def cluster_tuples(self, matched_tuples):
        """
        Single-Pass Clustering
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        count = 0
        for t in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(t, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

    def write_relationships_to_disk(self):
        print "\nWriting extracted relationships to disk"
        f_output = open("relationships.txt", "w")
        tmp = sorted(self.candidate_tuples.keys(), reverse=True)
        try:
            for t in tmp:
                f_output.write(
                    "instance: " + t.e1.encode("utf8") + '\t' + t.e2.encode("utf8") + '\tscore:' + str(t.confidence) +
                    '\n')
                f_output.write("sentence: " + t.sentence.encode("utf8") + '\n')
                f_output.write("pattern_bef: " + t.bef_words.encode("utf8") + '\n')
                f_output.write("pattern_bet: " + t.bet_words.encode("utf8") + '\n')
                f_output.write("pattern_aft: " + t.aft_words.encode("utf8") + '\n')
                if t.passive_voice is False:
                    f_output.write("passive voice: False\n")
                elif t.passive_voice is True:
                    f_output.write("passive voice: True\n")
                f_output.write("\n")
            f_output.close()
        except Exception, e:
            print e
            sys.exit(1)

    def init_bootstrap(self, tuples):
        """
        starts a bootstrap iteration
        """
        if tuples is not None:
            f = open(tuples, "r")
            print "Loading pre-processed sentences", tuples
            self.processed_tuples = cPickle.load(f)
            f.close()
            print len(self.processed_tuples), "tuples loaded"

        self.curr_iteration = 0
        while self.curr_iteration <= self.config.number_iterations:
            print "=========================================="
            print "\nStarting iteration", self.curr_iteration
            print "\nLooking for seed matches of:"
            for s in self.config.positive_seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            count_matches, matched_tuples = self.match_seeds_tuples()

            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)

            else:
                print "\nNumber of seed matches found"
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print t[0][0], '\t', t[0][1], t[1]
                print "\n", len(matched_tuples), "tuples matched"

                # Cluster the matched instances: generate patterns
                print "\nClustering matched instances to generate patterns"
                if len(self.patterns) == 0:
                    self.cluster_tuples(matched_tuples)
                    # Eliminate patterns supported by less than 'min_pattern_support' tuples
                    new_patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
                    self.patterns = new_patterns

                else:
                    # Paralelize single-pass clustering
                    # Each tuple must be compared with each extraction pattern
                    # Map:
                    # - Divide the tuples into smaller lists, accordingly to the number of CPUs
                    # - Pass to each CPU a sublist of tuples and all the patterns, comparision is done by each CPU
                    # Merge:
                    # - Each CPU sends to the father process the updated patterns and new patterns
                    # - Merge patterns based on Pattern_id
                    # - Cluster new created patterns with single-pass clustering

                    # make a copy of the extraction patterns to be passed to each
                    patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                    chunks = [list() for _ in range(self.num_cpus)]
                    n_tuples_per_child = int(math.ceil(float(len(matched_tuples)) / self.num_cpus))

                    print "\n#CPUS", self.num_cpus, '\t', "Tuples per CPU", n_tuples_per_child
                    chunk_n = 0
                    chunck_begin = 0
                    chunck_end = n_tuples_per_child
                    while chunk_n < self.num_cpus:
                        chunks[chunk_n] = matched_tuples[chunck_begin:chunck_end]
                        chunck_begin = chunck_end
                        chunck_end += n_tuples_per_child
                        chunk_n += 1

                    count = 0
                    for c in chunks:
                        print "CPU_"+str(count), len(c), "Patterns", len(patterns[count])
                        count += 1

                    pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
                    processes = [multiprocessing.Process(target=self.cluster_tuples_parallel, args=(patterns[i],
                                                                                                    chunks[i],
                                                                                                    pipes[i][1]))
                                 for i in range(self.num_cpus)]

                    print "\nRunning", len(processes), " processes"
                    for proc in processes:
                        proc.start()

                    # Receive and agregate all patterns by 'pattern_id'
                    # New created patterns (clusters) arae aggregated by single-pass clustering between patterns
                    child_patterns = list()
                    for i in range(len(pipes)):
                        data = pipes[i][0].recv()
                        patterns = data[1]
                        for p_updated in patterns:
                            for p_original in self.patterns:
                                if p_original.id == p_updated.id:
                                    p_original.tuples.update(p_updated.tuples)
                            child_patterns.append(p_updated)

                    for proc in processes:
                        proc.join()

                    print len(child_patterns), "new created patterns"

                    # merge/aggregate similar patterns generated by the child processes
                    # start comparing smaller ones with smaller ones and merging
                    child_patterns.sort(key=lambda y: len(y.tuples), reverse=False)

                    count = 0
                    new_list = child_patterns[:1]
                    child_patterns = child_patterns[1:]
                    for p1 in child_patterns:

                        print "\nNew Patterns", len(child_patterns), "Processed", count
                        print "New List", len(new_list)
                        print "Pattern:", p1.id, "Tuples:", len(p1.tuples)

                        max_similarity = 0
                        max_similarity_cluster = None
                        for p2 in new_list:
                            score = self.similarity_cluster(p1, p2)
                            if score > max_similarity:
                                max_similarity = score
                                max_similarity_cluster = p2

                        if max_similarity >= self.config.threshold_similarity:
                            print "Merging two patterns"
                            for t in p1.tuples:
                                max_similarity_cluster.tuples.add(t)
                        else:
                            new_list.append(p1)
                        count += 1

                    for p in new_list:
                        self.patterns.append(p)

                if self.curr_iteration == 0 and len(self.patterns) == 0:
                    print "No patterns generated"
                    sys.exit(0)

                print "\n", len(self.patterns), "patterns generated"

                # merge equal tuples inside patterns to make less comparisions in collecting instances
                for p in self.patterns:
                    # if only BET context is being used, merge only based on BET contexts
                    if self.config.alpha == 0 and self.config.gamma == 0:
                        p.merge_all_tuples_bet()
                    else:
                        p.merge_all_tuples()

                if PRINT_PATTERNS is True:
                    print "\nPatterns:"
                    for p in self.patterns:
                        print p.id
                        for t in p.tuples:
                            print "BEF", t.bef_words
                            print "BET", t.bet_words
                            print "AFT", t.aft_words
                            print "========"
                            print "\n"

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern
                # and store each pattern that has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns that extracted it
                # each with an associated degree of match.
                print "Number of tuples to be analyzed:", len(self.processed_tuples)

                print "\nCollecting instances based on extraction patterns"
                # create copies of generated extraction patterns to be passed to each process
                patterns = [list(self.patterns) for _ in range(self.num_cpus)]

                # copy all tuples into a Queue shared by all processes
                manager = multiprocessing.Manager()
                queue = manager.Queue()
                for t in self.processed_tuples:
                    queue.put(t)

                # each distinct process receives as arguments:
                #   - a list, copy of all the original extraction patterns
                #   - a Queue of the tuples
                #   - a pipe to return the collected tuples and updated patterns to the parent process

                pipes = [multiprocessing.Pipe(False) for _ in range(self.num_cpus)]
                processes = [multiprocessing.Process(target=self.find_instances, args=(patterns[i], queue, pipes[i][1]))
                             for i in range(self.num_cpus)]

                print "Running", len(processes), " processes"
                for proc in processes:
                    proc.start()

                # structures to store each process altered patterns and collected tuples
                patterns_updated = list()
                collected_tuples = list()

                for i in range(len(pipes)):
                    data = pipes[i][0].recv()
                    child_pid = data[0]
                    patterns = data[1]
                    tuples = data[2]
                    print child_pid, "patterns", len(patterns), "tuples", len(tuples)
                    patterns_updated.extend(patterns)
                    collected_tuples.extend(tuples)

                for proc in processes:
                    proc.join()

                # Extraction patterns aggregation happens here:
                for p_updated in patterns_updated:
                    for p_original in self.patterns:
                        if p_original.id == p_updated.id:
                            p_original.positive += p_updated.positive
                            p_original.negative += p_updated.negative
                            p_original.unknown += p_updated.unknown

                # Index the patterns in an hashtable for later use
                for p in self.patterns:
                    self.patterns_index[p.id] = p

                # Candidate tuples aggregation happens here:
                print "Collecting generated candidate tuples"
                for e in collected_tuples:
                    t = e[0]
                    pattern_best = e[1]
                    sim_best = e[2]

                    # if this tuple was already extracted, check if this extraction pattern is already associated
                    # with it, if not, associate this pattern with it and similarity score
                    if t in self.candidate_tuples:
                        t_patterns = self.candidate_tuples[t]
                        if t_patterns is not None:
                            if pattern_best not in [x[0] for x in t_patterns]:
                                self.candidate_tuples[t].append((self.patterns_index[pattern_best.id], sim_best))

                    # If this tuple was not extracted before, associate this pattern with the instance
                    # and the similarity score
                    else:
                        self.candidate_tuples[t].append((self.patterns_index[pattern_best.id], sim_best))

                # update all patterns confidence
                for p in self.patterns:
                    p.update_confidence(self.config)

                if PRINT_PATTERNS is True:
                    print "\nPatterns:"
                    for p in self.patterns:
                        print p.id
                        for t in p.tuples:
                            print "BEF", t.bef_words
                            print "BET", t.bet_words
                            print "AFT", t.aft_words
                            print "========"
                        print "Positive", p.positive
                        print "Negative", p.negative
                        print "Unknown", p.unknown
                        print "Tuples", len(p.tuples)
                        print "Pattern Confidence", p.confidence
                        print "\n"

                # update tuple confidence based on patterns confidence
                print "\n\nCalculating tuples confidence"
                for t in self.candidate_tuples.keys():
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                    # use past confidence values to calculate new tuple confidence
                    # if parameter Wupdt < 0.5 the system trusts new examples less on each iteration
                    # which will lead to more conservative patterns and have a damping effect.
                    if self.curr_iteration > 0:
                        t.confidence = t.confidence * self.config.wUpdt + t.confidence_old * (1 - self.config.wUpdt)

                # sort tuples by confidence and print
                if PRINT_TUPLES is True:
                    extracted_tuples = self.candidate_tuples.keys()
                    tuples_sorted = sorted(extracted_tuples, key=lambda tl: tl.confidence, reverse=True)
                    for t in tuples_sorted:
                        print t.sentence
                        print t.e1, t.e2
                        print t.confidence
                        print "\n"

                # update seed set of tuples to use in next iteration
                # seeds = { T | conf(T) > instance_confidance }
                print "Adding tuples to seed with confidence >=" + str(self.config.instance_confidance)
                for t in self.candidate_tuples.keys():
                    if t.confidence >= self.config.instance_confidance:
                        seed = Seed(t.e1, t.e2)
                        self.config.positive_seed_tuples.add(seed)

                # increment the number of iterations
                self.curr_iteration += 1

        self.write_relationships_to_disk()

    def similarity_cluster(self, p1, p2):
        count = 0
        score = 0
        if self.config.alpha == 0 and self.config.gamma == 0:
            p1.merge_all_tuples_bet()
            p2.merge_all_tuples_bet()
            for v_bet1 in p1.bet_uniques:
                for v_bet2 in p2.bet_uniques:
                    if v_bet1 is not None and v_bet2 is not None:
                        score += dot(matutils.unitvec(asarray(v_bet1)), matutils.unitvec(asarray(v_bet2)))
                        count += 1
        else:
            for t1 in p1.tuples:
                for t2 in p2.tuples:
                    score += self.similarity_3_contexts(t1, t2)
                    count += 1

        return float(score) / float(count)

    def find_instances(self, patterns, instances, child_conn):
        updated_patterns = list()
        candidate_tuples = list()
        while True:
            try:
                t = instances.get_nowait()
                if instances.qsize() % 500 == 0:
                    sys.stdout.write(str(multiprocessing.current_process()) +
                                     " Instances to process: "+str(instances.qsize())+'\n')
                    sys.stdout.flush()

                # measure similarity towards every extraction pattern
                sim_best = 0
                pattern_best = None
                for p in patterns:
                    accept, score = self.similarity_all(t, p)
                    if accept is True:
                        p.update_selectivity(t, self.config)
                        if score > sim_best:
                            sim_best = score
                            pattern_best = p

                # if its above a threshold associated the pattern with it
                if sim_best >= self.config.threshold_similarity:
                    candidate_tuples.append((t, pattern_best, sim_best))

            except Queue.Empty:
                print multiprocessing.current_process(), "Queue is Empty"
                for p in patterns:
                    updated_patterns.append(p)
                pid = multiprocessing.current_process().pid
                child_conn.send((pid, updated_patterns, candidate_tuples))
                break

    def cluster_tuples_parallel(self, patterns, matched_tuples, child_conn):
        updated_patterns = list(patterns)
        count = 0
        for t in matched_tuples:
            count += 1
            if count % 500 == 0:
                print multiprocessing.current_process(), count, "tuples processed"

            # Go through all patterns(clusters of tuples) and find the one with the highest similarity score
            max_similarity = 0
            max_similarity_cluster_index = 0
            for i in range(0, len(updated_patterns)):
                extraction_pattern = updated_patterns[i]
                accept, score = self.similarity_all(t, extraction_pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                updated_patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                updated_patterns[max_similarity_cluster_index].add_tuple(t)

        # Eliminate clusters with two or less patterns
        new_patterns = [p for p in updated_patterns if len(p.tuples) > 5]
        pid = multiprocessing.current_process().pid
        child_conn.send((pid, new_patterns))


def main():
    if len(sys.argv) != 8:
        print "\nBREDS.py paramters.cfg sentences_file seeds_file_positive seeds_file_negative similarity_threshold" \
              " confidance_threshold #cpus_to_use\n"
        sys.exit(0)
    else:
        configuration = sys.argv[1]
        sentences_file = sys.argv[2]
        seeds_file = sys.argv[3]
        negative_seeds = sys.argv[4]
        similarity = sys.argv[5]        # threshold similarity for clustering/extracting instances
        confidance = sys.argv[6]        # confidence threshold of an instance to used as seed
        num_cores = int(sys.argv[7])    # number of parallel jobs to launch
        breads = BREDS(configuration, seeds_file, negative_seeds, float(similarity), float(confidance), num_cores)
        if sentences_file.endswith('.pkl'):
            breads.init_bootstrap(tuples=sentences_file)
        else:
            breads.generate_tuples(sentences_file)
            breads.init_bootstrap(tuples=None)


if __name__ == "__main__":
    main()