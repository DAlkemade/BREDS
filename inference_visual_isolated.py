import fileinput
import logging
from argparse import ArgumentParser
from typing import List, Set
import networkx as nx
from logging_setup_dla.logging import set_up_root_logger

from breds.breds_inference import find_similar_words
from breds.config import Config
from breds.visual import VisualConfig
from datetime import datetime
import os
import tqdm

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)

class Pair:
    def __init__(self, e1, e2):
        self.e2 = e2
        self.e1 = e1
        self.larger = None


class VisualPropagation:
    def __init__(self, cooccurrence_graph, config: Config, max_path_length: int = 3):
        self.cooccurrence_graph = cooccurrence_graph

    def compare_pairs(self, pairs: Set[Pair]):
        for pair in pairs:
            self.compare_pair(pair)

    def compare_pair(self, pair: Pair) -> None:
        """Use propagation to compare two objects visually.

        Finds all paths of lenght <= self.max_path_length between the two objects and computes
        the size comparisons on all edges on the paths. Then uses the fraction of paths indicating one object being
        larger as the confidence of that being true.
        Results are saved in-place in the Pair object.
        :param pair: pair to be predicted
        """
        pass

def main():
    parser = ArgumentParser()
    parser.add_argument('--test_pairs', type=str, required=True)
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
    test_pairs_fname = args.test_pairs



    # TODO check whether the objects aren't in the bootstrapped objects
    visual_config = VisualConfig(args.vg_objects, args.vg_objects_anchors)
    config = Config(args.configuration, args.seeds_file, args.negative_seeds, args.similarity, args.confidence,
                    args.objects, visual_config)

    visual_config = config.visual_config
    objects = list(visual_config.entity_to_synsets.keys())
    G = nx.Graph()
    G.add_nodes_from(objects)
    logger.info(f'Number of nodes: {G.number_of_nodes()}')
    for object1 in tqdm.tqdm(objects):
        synsets1 = visual_config.entity_to_synsets[object1]
        s1 = synsets1[0] # TODO this is bad, do for all synsets
        for object2 in objects:

            synsets2 = visual_config.entity_to_synsets[object2]
            s2 = synsets2[0]  # TODO this is bad, do for all synsets
            cooccurrences = len(visual_config.comparer.find_cooccurrences(s1, s2))
            if cooccurrences > 0:
                G.add_edge(object1, object2, weight=cooccurrences)
    nr_edges = G.number_of_edges()
    max_edges = G.number_of_nodes()**2
    logger.info(f'Number of edges: {nr_edges} (sparsity: {nr_edges/max_edges})')


    test_pairs = set()
    for line in fileinput.input(test_pairs_fname):
        split = line.split(',')
        test_pairs.add(Pair(split[0], split[1]))




if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise