import math
from dataclasses import dataclass

from discopygal.bindings import Pol2


@dataclass
class Partition:
    """One cell of the free-space decomposition.

    density is the disc-packing lower bound `max(1, floor(area / (pi*(2r)^2)))`
    and is also used as the HLG edge capacity. complexity is `1/q` for the
    isoperimetric quotient `q = 4*pi*area/perimeter^2` (q=1 for a circle,
    smaller for thinner / arc-carved shapes).
    """
    polygon: Pol2.Polygon_2
    robot_radius: float = 0.5
    density: int = 0
    complexity: float = 0.0

    def __post_init__(self) -> None:
        cell_unit = math.pi * (2.0 * self.robot_radius) ** 2
        # CGAL polygon area is signed; take the magnitude.
        area = abs(self.polygon.area().to_double())
        self.density = max(1, math.floor(area / cell_unit))
        self.complexity = self._compute_complexity(area)

    def update_density(self, new_density: int) -> None:
        """Override the auto-computed density (used by the topological cap
        and the density-retry loop)."""
        self.density = new_density

    def _compute_complexity(self, area: float) -> float:
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
