"""Per-cell joint k-robot PRM.

Step 4 of ``plan.tex``. For every trapezoidal ``Partition`` we sample a
probabilistic roadmap whose nodes are *joint* configurations of up to ``k``
robots simultaneously occupying the cell, where ``k = partition.density``.

Geometry
--------
The cell is a convex polygon (a trapezoid coming out of the vertical
decomposition). For convex domains, two configurations whose individual
robot positions both lie inside the cell are also connected by a straight
line that stays inside the cell. That collapses the per-robot collision
check inside the cell to nothing — the only remaining concern is *robot–
robot* collision in the **joint** space.

A joint configuration ``c = (p_1, …, p_k)`` is **valid** when

    ∀ i ≠ j :  ‖p_i − p_j‖ ≥ 2r          (pairwise non-overlap)
    ∀ i     :  p_i ∈ cell                (containment)

Two valid joint configurations ``c`` and ``c'`` are **connected** by a
straight-line steer in joint space if every interpolated configuration
along the segment is also valid. Because the cell is convex, individual
``p_i(t) = (1−t)·p_i + t·p'_i`` stays inside, so we only re-check the
pairwise separation condition along a discretised sweep.

Boundary ports
--------------
The high-level graph (step 5) wires cells together at the midpoint of every
shared edge between adjacent cells — those midpoints are the **ports**. To
let the multi-commodity flow extract a real motion plan we need each port
to correspond to at least one node in the cell roadmap. We construct a
*port joint config* by pinning the first robot at the port midpoint and
sampling the other ``k − 1`` robots inside the cell so that pairwise
separation holds. The "first robot" slot is bookkeeping only; the high-level
graph does not care which slot a particular agent occupies — it cares that
*some* node in the cell graph realises the port.

Output
------
A ``CellRoadmap`` containing
- ``partition``: the source cell,
- ``graph``: a ``networkx.Graph`` whose nodes are joint configurations and
  whose edge weights are the maximum single-robot travel along the steer,
- ``port_nodes``: ``{boundary_id → joint_config}`` so the high-level graph
  can hand the cell a (entry_port, exit_port) pair and ask the roadmap for
  a path between the two anchor nodes.

Limitations
-----------
- Sampling is rejection-based and may fail to populate roadmaps for very
  tight cells whose ``density`` is optimistic. The fix when this becomes a
  problem is to lower the density formula in ``Partition`` or raise
  ``num_samples``.
- We approximate the swept-volume collision check with a fixed number of
  steps (``DEFAULT_STEER_STEPS``). Increase if cells are large relative to
  the robot radius.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side

from discopygal.bindings import FT, Pol2, Point_2

from partition import Partition


# A joint configuration is a tuple of (x, y) tuples — one per robot slot.
JointConfig = Tuple[Tuple[float, float], ...]

DEFAULT_STEER_STEPS = 10
DEFAULT_NUM_SAMPLES = 30
DEFAULT_K_NEAREST = 6


# ---------------------------------------------------------------------------
# Cell geometry helpers
# ---------------------------------------------------------------------------

def _polygon_bbox(poly: Pol2.Polygon_2) -> Tuple[float, float, float, float]:
    xs = [v.x().to_double() for v in poly.vertices()]
    ys = [v.y().to_double() for v in poly.vertices()]
    return min(xs), max(xs), min(ys), max(ys)


def _point_inside_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
    """Closed (boundary-inclusive) containment for a CGAL polygon."""
    side = poly.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _pairwise_separated(points: List[Tuple[float, float]], min_dist_sq: float) -> bool:
    for i in range(len(points)):
        xi, yi = points[i]
        for j in range(i + 1, len(points)):
            dx = xi - points[j][0]
            dy = yi - points[j][1]
            if dx * dx + dy * dy < min_dist_sq:
                return False
    return True


def _config_distance(c1: JointConfig, c2: JointConfig) -> float:
    """Maximum single-robot displacement between two joint configurations.

    Used as the steer cost — the joint motion takes time proportional to
    the slowest robot, so the longest individual displacement is the right
    proxy.
    """
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(c1, c2))


def _steer_collision_free(
    c1: JointConfig,
    c2: JointConfig,
    robot_radius: float,
    steps: int = DEFAULT_STEER_STEPS,
) -> bool:
    """Check pairwise separation along a straight-line joint-space steer.

    Containment is implicit because both endpoints are inside the same
    convex cell.
    """
    min_sq = (2.0 * robot_radius) ** 2
    k = len(c1)
    for s in range(steps + 1):
        t = s / steps
        cur = [
            (
                c1[i][0] + t * (c2[i][0] - c1[i][0]),
                c1[i][1] + t * (c2[i][1] - c1[i][1]),
            )
            for i in range(k)
        ]
        if not _pairwise_separated(cur, min_sq):
            return False
    return True


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_joint_config(
    poly: Pol2.Polygon_2,
    k: int,
    robot_radius: float,
    rng: random.Random,
    pinned: Optional[List[Tuple[float, float]]] = None,
    max_tries: int = 500,
) -> Optional[JointConfig]:
    """Rejection-sample a valid joint configuration of ``k`` robots.

    If ``pinned`` is given those positions are fixed (already validated by
    the caller) and only the remaining slots are sampled.
    """
    if k == 0:
        return tuple(pinned or ())

    min_sq = (2.0 * robot_radius) ** 2
    minx, maxx, miny, maxy = _polygon_bbox(poly)
    points: List[Tuple[float, float]] = list(pinned or [])
    if not _pairwise_separated(points, min_sq):
        return None

    tries = 0
    while len(points) < k and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if not _point_inside_polygon(poly, x, y):
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sq for px, py in points):
            points.append((x, y))
    if len(points) < k:
        return None
    return tuple(points)


# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------

@dataclass
class CellRoadmap:
    partition: Partition
    graph: nx.Graph
    port_nodes: Dict[int, JointConfig]


def build_cell_roadmap(
    partition: Partition,
    boundary_ports: Dict[int, Tuple[float, float]],
    num_samples: int = DEFAULT_NUM_SAMPLES,
    k_nearest: int = DEFAULT_K_NEAREST,
    steer_steps: int = DEFAULT_STEER_STEPS,
    seed: Optional[int] = None,
) -> CellRoadmap:
    """Build a joint-space PRM inside a single trapezoidal cell.

    Args:
        partition: cell to roadmap.
        boundary_ports: ``{boundary_id: (x, y)}`` — midpoints of shared
            edges with neighbour cells. Each becomes an *anchor* joint
            configuration whose first robot sits on the port and whose
            remaining robots are sampled away from it.
        num_samples: number of random interior joint configurations to draw.
        k_nearest: each sample tries to connect to its ``k_nearest`` joint-
            distance nearest neighbours; collision-free attempts become
            roadmap edges.
        steer_steps: discretisation of the swept-volume collision check.
        seed: optional RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    poly = partition.polygon
    k = max(1, partition.density)
    r = partition.robot_radius

    graph = nx.Graph()
    samples: List[JointConfig] = []

    # Random interior samples
    for _ in range(num_samples):
        cfg = _sample_joint_config(poly, k, r, rng)
        if cfg is None:
            continue
        if cfg not in graph:
            graph.add_node(cfg)
            samples.append(cfg)

    # Boundary-port anchor configs
    port_nodes: Dict[int, JointConfig] = {}
    for boundary_id, (px, py) in boundary_ports.items():
        if not _point_inside_polygon(poly, px, py):
            continue
        cfg = _sample_joint_config(
            poly, k, r, rng, pinned=[(px, py)], max_tries=500
        )
        if cfg is None:
            continue
        if cfg not in graph:
            graph.add_node(cfg)
            samples.append(cfg)
        port_nodes[boundary_id] = cfg

    # k-NN connection inside the joint space
    for i, ci in enumerate(samples):
        candidates = sorted(
            (
                (_config_distance(ci, cj), j)
                for j, cj in enumerate(samples)
                if j != i
            ),
            key=lambda pair: pair[0],
        )[:k_nearest]
        for distance, j in candidates:
            cj = samples[j]
            if graph.has_edge(ci, cj):
                continue
            if _steer_collision_free(ci, cj, r, steer_steps):
                graph.add_edge(ci, cj, weight=distance)

    return CellRoadmap(partition=partition, graph=graph, port_nodes=port_nodes)
