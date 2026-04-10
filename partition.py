"""A single decomposed cell of the free space.

A ``Partition`` wraps a CGAL polygon (the cell shape) and exposes a *density*:
the maximum number of disc robots that may simultaneously occupy the cell.

Density formula
---------------
We use the disc-packing lower bound. A disc robot of radius ``r`` is
contained in its 2r x 2r axis-aligned bounding square. ``ceil``-packing those
squares into the cell area gives::

    density = max(1, ⌊ area / (π · (2r)²) ⌋)

We use ``π · (2r)²`` rather than ``(2r)²`` to be conservative — this leaves
slack for the curved swept volume of moving discs and avoids over-packing.
The ``max(1, …)`` floor guarantees that *some* robot can transit even very
small cells (otherwise narrow corridors become uncrossable).
"""

import math
from dataclasses import dataclass

from discopygal.bindings import Pol2


@dataclass
class Partition:
    polygon: Pol2.Polygon_2
    robot_radius: float = 0.5
    density: int = 1

    def __post_init__(self) -> None:
        cell_unit = math.pi * (2.0 * self.robot_radius) ** 2
        area = self.polygon.area().to_double()
        # Area of a CGAL polygon is signed; take the magnitude.
        self.density: int = max(1, math.floor(abs(area) / cell_unit))


    def update_density(self, new_density: int) -> None:
        self.density = new_density