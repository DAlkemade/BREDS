from typing import List

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"


class Seed(object):
    def __init__(self, _e1, _e2: float):
        self.e1 = _e1
        self.sizes: List[float] = [_e2]

    def add_size(self, size: float):
        self.sizes.append(size)

    def __hash__(self):
        return hash(self.e1) ^ hash(self.sizes)

    def __eq__(self, other):
        return self.e1 == other.e1 and self.sizes == other.e2
