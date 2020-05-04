from typing import List, Dict

import numpy as np
from visual_size_comparison.compare import Comparer
from visual_size_comparison.objects import load_images_index, index_objects

import pandas as pd


class VisualConfig:
    def __init__(self, vg_objects, vg_objects_anchors):
        images = load_images_index(vg_objects)
        objects_lookup = index_objects(images)

        self.comparer = Comparer(objects_lookup, images)

        test_objects_df = pd.read_csv(vg_objects_anchors)
        self.test_objects = list(test_objects_df.itertuples(index=False))
        self.entity_to_synsets: Dict[str, List[str]] = dict()
        self.fill_synset_mapping(list(objects_lookup.keys()))

    def fill_synset_mapping(self, synsets: List[str]):
        for synset in synsets:
            name_raw = synset.split('.')[0]
            try:
                self.entity_to_synsets[name_raw].append(synset)
            except KeyError:
                self.entity_to_synsets[name_raw] = [synset]

def check_tuple_with_visuals(config: VisualConfig, entity, candidate_size) -> List[bool]:
    corresponds_to_visual_anchors = []

    try:
        synsets = config.entity_to_synsets[entity]
    except KeyError:
        # logger.info(f'Entity {tuple_e1} not in visuals; not using for confidence')
        # TODO log which entities weren't in visuals
        return corresponds_to_visual_anchors

    for synset in synsets:
        for visual_anchor in config.test_objects:
            comp = config.comparer.compare(synset, visual_anchor.object)
            if len(comp) >= 1:
                res = np.mean(comp)
                correct: bool = (candidate_size > visual_anchor.size) == (res > 1.)
                corresponds_to_visual_anchors.append(correct)


    return corresponds_to_visual_anchors