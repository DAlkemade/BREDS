from __future__ import division
import lucene
# noinspection PyUnresolvedReferences
from java.nio.file import Paths
# noinspection PyUnresolvedReferences
from org.apache.lucene.search import IndexSearcher, PhraseQuery, RegexpQuery
# noinspection PyUnresolvedReferences
from org.apache.lucene.search.spans import SpanMultiTermQueryWrapper, SpanNearQuery
# noinspection PyUnresolvedReferences
from org.apache.lucene.index import DirectoryReader, Term
# noinspection PyUnresolvedReferences
from org.apache.lucene.store import FSDirectory
# noinspection PyUnresolvedReferences
from org.apache.lucene.queryparser.classic import QueryParser
# noinspection PyUnresolvedReferences
from org.apache.lucene.analysis.standard import StandardAnalyzer

if __name__ == "__main__":
     # noinspection PyUnresolvedReferences
     lucene.initVM(initialheap='32m', maxheap='4G')
     file = Paths.get("D:\GitHubD\BREDS\wiki_text_index\WIKI_TEXT")
     dir = FSDirectory.open(file)
     reader = DirectoryReader.open(dir)
     searcher = IndexSearcher(reader)

     term = Term("contents", "tiger")
     print(f'Tiger frequency: {reader.totalTermFreq(term)}')

     q_regex = RegexpQuery(Term("contents", "[0-9]+\.?[0-9]*"))
     print(f'regex results: {searcher.search(q_regex,1000000).totalHits}')

     span1 = SpanMultiTermQueryWrapper(q_regex)
     span2 = SpanMultiTermQueryWrapper(RegexpQuery(Term("contents", "tiger")))
     spannearquery = SpanNearQuery([span1, span2], 20, True)
     print(f'spanquery results: {searcher.search(spannearquery, 1000000).totalHits}')

     parser = QueryParser('contents', StandardAnalyzer())
     q = parser.parse('"tiger leopard"')
     print(q)  # prints contents:"tiger leopard"
     print(searcher.search(q, 10000000).totalHits)

     phrase_query = PhraseQuery(10, 'contents', 'tiger leopard')
     print(phrase_query)
     print(searcher.search(phrase_query, 10000000).totalHits)

     parser = QueryParser('contents', StandardAnalyzer())
     q = parser.parse('"tiger leopard"~10')
     print(q) # prints contents:"tiger leopard"~10
     print(searcher.search(q, 10000000).totalHits)



     for i in range(0,reader.numDocs()):
          doc = reader.document(i)
          text = doc.get("contents")
          articleID  = doc.get("articleID")
          # Do your pattern matching and record patterns for document
     articleID

     reader.close()