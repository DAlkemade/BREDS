import logging
from typing import List, Set

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

logger = logging.getLogger(__name__)

class Seed(object):
    def __init__(self, _e1, _e2: float):
        self.e1 = _e1
        self.sizes: Set[float] = [_e2]
        logger.info(
            f"Created new seed {_e1} {_e2}")

    def add_size(self, size: float):
        if not size in self.sizes:
            logger.info(f'Add new size to existing seed: {self.e1} {size}')
        self.sizes.add(size)

    def __hash__(self):
        return hash(self.e1) ^ hash(self.sizes)

    def __eq__(self, other):
        return self.e1 == other.e1 and self.sizes == other.e2
