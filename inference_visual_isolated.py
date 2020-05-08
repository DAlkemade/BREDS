import fileinput
from argparse import ArgumentParser
from typing import List

from breds.breds_inference import find_similar_words
from breds.config import Config
from breds.visual import VisualConfig

class Pair:
    def __init__(self, e1, e2, config: Config):
        self.config = config
        self.e2 = e2
        self.e1 = e1
        self.larger = None


class VisualPropagation:
    def __init__(self, cooccurrence_graph, max_path_length: int = 3):
        self.cooccurrence_graph = cooccurrence_graph

    def compare_pairs(self, pairs: List[Pair]):
        for pair in pairs:
            self.compare_pair(pair)

    def compare_pair(self, pair: Pair) -> None:
        """Use propagation to compare two objects visually.

        Finds all paths of lenght <= self.max_path_length between the two objects and computes
        the size comparisons on all edges on the paths.
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
    test_pairs = None  # TODO

    cache_fname = 'inference_cache.pkl'
    similar_words = find_similar_words(config, unseen_objects)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise