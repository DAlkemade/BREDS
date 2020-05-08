import fileinput
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from queue import Queue
from typing import Set

import networkx as nx
import tqdm
from logging_setup_dla.logging import set_up_root_logger

from breds.config import Config
from breds.visual import VisualConfig

set_up_root_logger(f'INFERENCE_VISUAL_{datetime.now().strftime("%d%m%Y%H%M%S")}', os.path.join(os.getcwd(), 'logs'))
logger = logging.getLogger(__name__)


class Pair:
    def __init__(self, e1, e2):
        self.e2 = e2
        self.e1 = e1
        self.larger = None

    def both_in_list(self, objects: list):
        return self.e1 in objects and self.e2 in objects


class VisualPropagation:
    def __init__(self, cooccurrence_graph: nx.Graph, config: Config, max_path_length: int = 3):
        self.config = config
        self.max_path_length: int = max_path_length
        self.cooccurrence_graph: nx.Graph = cooccurrence_graph

    def find_paths(self, pair: Pair):
        good_paths = list()
        queue = Queue()
        queue.put([pair.e1])

        while not queue.empty():
            path = queue.get()
            if path[-1] == pair.e2:
                good_paths.append(path)
                continue
            if len(path) >= self.max_path_length:
                continue
            last_node = path[-1]
            neighbours = self.cooccurrence_graph.neighbors(last_node)
            for n in neighbours:
                if n in path:
                    continue
                queue.put(path + [n])

        logger.info(f'Found paths: {good_paths}')
        return good_paths

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
        s1 = synsets1[0]  # TODO this is bad, do for all synsets
        for object2 in objects:

            synsets2 = visual_config.entity_to_synsets[object2]
            s2 = synsets2[0]  # TODO this is bad, do for all synsets
            cooccurrences = len(visual_config.comparer.find_cooccurrences(s1, s2))
            if cooccurrences > 0:
                G.add_edge(object1, object2, weight=cooccurrences)
    nr_edges = G.number_of_edges()
    max_edges = G.number_of_nodes() ** 2
    logger.info(f'Number of edges: {nr_edges} (sparsity: {nr_edges / max_edges})')

    test_pairs: Set[Pair] = set()
    for line in fileinput.input(test_pairs_fname):
        split = line.split(',')
        test_pairs.add(Pair(split[0], split[1]))

    prop = VisualPropagation(G, config)
    for test_pair in test_pairs:
        if test_pair.both_in_list(objects):
            prop.find_paths(test_pair)
        else:
            logger.info(f'{test_pair.e1} {test_pair.e2} not in VG')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
