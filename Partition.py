import math
from dataclasses import dataclass

from discopygal.bindings import Pol2


@dataclass
class Partition:
    polygon: Pol2

    def __post_init__(self):
        # TODO: add the maximum robot size to the calculation of the partition density - this should replace the 4.0 with the 
        self.density = math.floor(self.polygon.area().to_double() / 4.0)
