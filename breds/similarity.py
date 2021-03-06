from gensim import matutils
from numpy import dot

from breds.config import Weights
from breds.pattern import Pattern
from breds.tuple import Tuple


def similarity_3_contexts(p: Tuple, t: Tuple, weights: Weights):
    (bef, bet, aft) = (0, 0, 0)

    if t.bef_vector is not None and p.bef_vector is not None:
        bef = dot(matutils.unitvec(t.bef_vector), matutils.unitvec(p.bef_vector))

    if t.bet_vector is not None and p.bet_vector is not None:
        bet = dot(matutils.unitvec(t.bet_vector), matutils.unitvec(p.bet_vector))

    if t.aft_vector is not None and p.aft_vector is not None:
        aft = dot(matutils.unitvec(t.aft_vector), matutils.unitvec(p.aft_vector))

    return weights.alpha * bef + weights.beta * bet + weights.gamma * aft


def similarity_all(t: Tuple, extraction_pattern: Pattern, weights: Weights, threshold_similarity: float):
    # calculates the cosine similarity between all patterns part of a
    # cluster (i.e., extraction pattern) and the vector of a ReVerb pattern
    # extracted from a sentence;

    # returns the max similarity scores

    good = 0
    bad = 0
    max_similarity = 0

    for p in list(extraction_pattern.tuples):
        score = similarity_3_contexts(t, p, weights)
        if score > max_similarity:
            max_similarity = score
        if score >= threshold_similarity:
            good += 1
        else:
            bad += 1

    if good >= bad:
        return True, max_similarity
    else:
        return False, 0.0
