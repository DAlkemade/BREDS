from __future__ import division
import mmap
import lucene
from java.nio.file import Paths
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import FSDirectory

import tika.parser

if __name__ == "__main__":
     lucene.initVM(initialheap='32m', maxheap='4G')
     file = Paths.get("D:\GitHubD\BREDS\wiki_text_index\WIKI_TEXT")
     dir = FSDirectory.open(file)
     reader = DirectoryReader.open(dir)
     for i in range(0,reader.numDocs()):
          doc = reader.document(i)
          text = doc.get("contents")
          articleID  = doc.get("articleID")
          print(text)
          # Do your pattern matching and record patterns for document
     articleID

     reader.close()