"""A single decomposed cell of the free space.

A ``Partition`` wraps a CGAL polygon (the cell shape) and exposes two
scalars used later in the pipeline:

- ``density`` — maximum number of disc robots that may simultaneously
  occupy the cell (MCF edge capacity).
- ``complexity`` — shape-difficulty factor used by the staged solver to
  scale the joint-PRM sample count for this cell.

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

Complexity formula
------------------
``complexity = 1 / q`` where ``q = 4π · area / perimeter²`` is the
isoperimetric quotient. ``q = 1`` for a circle and shrinks toward ``0`` for
long/thin or arc-carved shapes, so ``complexity`` is ``1`` for round cells
and grows with shape difficulty. The staged solver multiplies the base
PRM sample count by ``min(4.0, √complexity)``.
"""

import math
from dataclasses import dataclass

from discopygal.bindings import Pol2


@dataclass
class Partition:
    polygon: Pol2.Polygon_2
    robot_radius: float = 0.5
    density: int = 1
    complexity: float = 1.0

    def __post_init__(self) -> None:
        cell_unit = math.pi * (2.0 * self.robot_radius) ** 2
        area = abs(self.polygon.area().to_double())
        # Area of a CGAL polygon is signed; take the magnitude.
        self.density: int = max(1, math.floor(area / cell_unit))
        self.complexity = self._compute_complexity(area)


    def update_density(self, new_density: int) -> None:
        self.density = new_density

    def _compute_complexity(self, area: float) -> float:
        """Isoperimetric-quotient-based shape difficulty score.

        Circle → ``1.0``. Long/thin or arc-carved → larger. Capped at the
        caller via ``min(4.0, √complexity)`` so a pathological cell does
        not crater runtime.
        """
        verts = list(self.polygon.vertices())
        perim = 0.0
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            perim += math.hypot(
                b.x().to_double() - a.x().to_double(),
                b.y().to_double() - a.y().to_double(),
            )
        if perim < 1e-9 or area < 1e-9:
            return 1.0
        q = max(1e-6, 4.0 * math.pi * area / (perim * perim))
        return 1.0 / q
