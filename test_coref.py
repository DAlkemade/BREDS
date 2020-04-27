import neuralcoref
import spacy

from parse_coref import parse_coref


def test_coref():
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    res = parse_coref(['A Lion is an animal. It is 3m long.', 'A mouse is an animal. It is .3m long.'], nlp, 'lion')
    assert 'A Lion is an animal. A Lion is 3m long.' in res
    assert 'A mouse is an animal. It is .3m long.' in res
