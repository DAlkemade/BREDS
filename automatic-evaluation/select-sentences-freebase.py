#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import codecs
import fileinput
import multiprocessing
import sys

from os import listdir
from os.path import isfile, join
from BREDS.Sentence import Sentence

MAX_TOKENS_AWAY = 6
MIN_TOKENS_AWAY = 1
CONTEXT_WINDOW = 2

manager = multiprocessing.Manager()


def load_yago_entities(data_file):
    entities = manager.dict()
    count = 0
    for line in fileinput.input(data_file):
        if line.startswith("#"):
            continue
        try:
            e1, r, e2 = line.split('\t')
            entities[e1] = 'dummy'
            entities[e2] = 'dummy'
            count += 1
            if count % 100000 == 0:
                print count, "processed"
        except Exception, e:
            print e
            print line
            sys.exit(0)
        fileinput.close()
    return entities


def load_dbpedia_entities(data_file):
    entities = manager.dict()
    count = 0
    for line in fileinput.input(data_file):
        if line.startswith("#"):
            continue
        try:
            e1, r, e2 = line.split('\t')
            entities[e1] = 'dummy'
            entities[e2] = 'dummy'
            count += 1
            if count % 100000 == 0:
                print count, "processed"
        except Exception, e:
            print e
            print line
            sys.exit(0)
        fileinput.close()
    return entities


def load_freebase_entities(directory):
    entities = manager.dict()
    onlyfiles = [data_file for data_file in listdir(directory) if isfile(join(directory, data_file))]
    print onlyfiles
    for data_file in onlyfiles:
        print "Processing", directory+data_file
        count = 0
        for line in fileinput.input(directory+data_file):
            if line.startswith("#"):
                continue
            try:
                e1, r, e2 = line.split('\t')
                entities[e1] = 'dummy'
                entities[e2] = 'dummy'
                count += 1
                if count % 100000 == 0:
                    print count, "processed"
            except Exception, e:
                print e
                print line
                sys.exit(0)
        fileinput.close()
    return entities


def load_sentences(data_file):
    sentences = manager.Queue()
    count = 0
    with codecs.open(data_file, 'rb', encoding='utf-8') as f:
        for line in f:
            sentences.put(line.strip())
            count += 1
            if count % 100000 == 0:
                print count, "processed"

        return sentences


def get_sentences(sentences, freebase, results):
    count = 0
    while True:
        sentence = sentences.get_nowait()
        discard = False
        count += 1
        s = Sentence(sentence.strip(), None, None, MAX_TOKENS_AWAY, MIN_TOKENS_AWAY, CONTEXT_WINDOW)
        for r in s.relationships:
            if r.between == " , " or r.between == " ( " or r.between == " ) ":
                discard = True
                break
            else:
                if r.ent1 not in freebase or r.ent2 not in freebase:
                    discard = True
                    break
        if len(s.relationships) == 0:
            discard = True
        if discard is False:
            results.append(sentence)

        if count % 50000 == 0:
            print multiprocessing.current_process(), "queue size", sentences.qsize()

        if sentences.empty is True:
            break


def main():
    # load freebase entities into a shared data structure
    freebase = load_freebase_entities(sys.argv[1])
    print len(freebase), " Freebase entities loaded"

    # load freebase entities into a shared data structure
    dbpedia = load_dbpedia_entities(sys.argv[2])
    print len(dbpedia), " DBpedia entities loaded"

    # load freebase entities into a shared data structure
    yago = load_yago_entities(sys.argv[2])
    print len(yago), " YAGO entities loaded"

    print "Loading sentences"
    sentences = load_sentences(sys.argv[2])
    print sentences.qsize(), " sentences loaded"

    print "Selecting sentences with entities in the KB"
    # launch different processes, each reads a sentence from AFP/APW news corpora
    # transform sentence into relationships, if both entities in relationship occur in freebase
    # select sentence
    num_cpus = multiprocessing.cpu_count()
    results = [manager.list() for _ in range(num_cpus)]

    processes = [multiprocessing.Process(target=get_sentences, args=(sentences, freebase, results[i]))
                 for i in range(num_cpus)]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    selected_sentences = set()
    for l in results:
        selected_sentences.update(l)

    print "Writing sentences to disk"
    f = open("sentences_matched_output.txt", "w")
    for s in selected_sentences:
        try:
            f.write(s+'\n')
        except Exception, e:
            print e
            print type(s)
            print s
    f.close()

if __name__ == "__main__":
    main()