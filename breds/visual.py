from typing import List

import numpy as np
from visual_size_comparison.config import VisualConfig


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