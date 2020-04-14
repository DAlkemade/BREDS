from __future__ import division

import logging

import lucene
# noinspection PyUnresolvedReferences
from java.nio.file import Paths
# noinspection PyUnresolvedReferences
from org.apache.lucene.analysis.standard import StandardAnalyzer
# noinspection PyUnresolvedReferences
from org.apache.lucene.index import DirectoryReader, Term
# noinspection PyUnresolvedReferences
from org.apache.lucene.queryparser.classic import QueryParser
# noinspection PyUnresolvedReferences
from org.apache.lucene.search import IndexSearcher, PhraseQuery
# noinspection PyUnresolvedReferences
from org.apache.lucene.store import FSDirectory


def find_all_text_occurrences(objects: list) -> (dict, DirectoryReader):
    docs_lookup = dict()
    # noinspection PyUnresolvedReferences
    lucene.initVM(initialheap='32m', maxheap='4G')
    file = Paths.get("D:\GitHubD\BREDS\wiki_text_index\WIKI_TEXT")
    dir = FSDirectory.open(file)
    reader = DirectoryReader.open(dir)
    searcher = IndexSearcher(reader)
    parser = QueryParser('contents', StandardAnalyzer())

    logging.warning('FOR MULTI-WORD OBJECTS, ALL DOCUMENTS WITH BOTH TERMS SEPARATELY WILL BE RETRIEVED')

    for object in objects:
        tokens = object.split(' ')

        doc_sets = []
        for token in tokens:
            q = parser.parse(f'"{token}"')
            # TODO maybe use minimum score
            topdocs = searcher.search(q, 99999999)
            results = set([topdoc.doc for topdoc in topdocs.scoreDocs])
            doc_sets.append(results)
        docs_lookup[object] = set.intersection(*doc_sets)


    return docs_lookup, reader
