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

from discopygal.bindings import FT, Pol2, Point_2, Segment_2

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


def _min_edge_distance(poly: Pol2.Polygon_2, x: float, y: float) -> float:
    """Signed distance from ``(x, y)`` to the nearest edge of a CCW polygon.

    Positive means inside, negative means outside.  For a convex CCW
    polygon this equals the inset depth — i.e. how far from the boundary
    the point sits.
    """
    verts = [(v.x().to_double(), v.y().to_double()) for v in poly.vertices()]
    n = len(verts)
    min_d = float("inf")
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.sqrt(ex * ex + ey * ey)
        if length < 1e-12:
            continue
        # Signed distance: positive = left of edge = inside for CCW
        d = (ex * (y - ay) - ey * (x - ax)) / length
        if d < min_d:
            min_d = d
    return min_d


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
    poly: Optional[Pol2.Polygon_2] = None,
    checker=None,
) -> bool:
    """Check pairwise separation along a straight-line joint-space steer.

    If ``poly`` is ``None`` containment is assumed (valid for convex cells
    from vertical decomposition). If ``poly`` is supplied, every
    interpolated robot position is additionally tested for polygon
    containment — required for the grid-only partition mode, whose cells
    can be non-convex where inflated obstacle boundaries carve notches.

    If ``checker`` is supplied (a discopygal ``ObjectCollisionDetection``
    instance), each robot's straight-line segment is additionally vetted
    via ``checker.is_edge_valid`` — the authoritative swept-disc vs scene-
    obstacle check that ``verify_paths`` itself uses. This catches cases
    where the polygon approximation of a curved inflated-obstacle boundary
    (arc-vs-chord gap) incorrectly claims blocked area as free.
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
        if poly is not None:
            for x, y in cur:
                if not _point_inside_polygon(poly, x, y):
                    return False
    if checker is not None:
        for i in range(k):
            # Degenerate zero-length segments (pinned holders) are always
            # safe — skip them to avoid CGAL degenerate-segment asserts.
            if c1[i] == c2[i]:
                continue
            seg = Segment_2(
                Point_2(FT(c1[i][0]), FT(c1[i][1])),
                Point_2(FT(c2[i][0]), FT(c2[i][1])),
            )
            if not checker.is_edge_valid(seg):
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
    max_tries_per_robot: int = 200,
    boundary_margin: float = 0.0,
    checker=None,
) -> Optional[JointConfig]:
    """Incrementally sample a valid joint configuration of ``k`` robots.

    Places robots one at a time. Each new robot must be inside the polygon,
    at least ``boundary_margin`` from every edge, and separated from all
    previously placed robots by at least ``2r``.

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

    while len(points) < k:
        placed = False
        for _ in range(max_tries_per_robot):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if not _point_inside_polygon(poly, x, y):
                continue
            if boundary_margin > 0 and _min_edge_distance(poly, x, y) < boundary_margin:
                continue
            if not all((x - px) ** 2 + (y - py) ** 2 >= min_sq for px, py in points):
                continue
            # Authoritative check against true inflated obstacles. Needed
            # for non-convex cells from grid-only partitioning where the
            # polygon approximates arc boundaries with chords.
            if checker is not None and not checker.is_point_valid(
                Point_2(FT(x), FT(y))
            ):
                continue
            points.append((x, y))
            placed = True
            break
        if not placed:
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

    The PRM operates in joint space with ``k = partition.density`` robots.
    The grid refinement in ``scene_partitioning`` ensures that density
    values are bounded by ``max_cell_density`` (default 8), keeping the
    joint-space dimension tractable for the incremental placement sampler.
    """
    rng = random.Random(seed)
    poly = partition.polygon
    k = max(1, partition.density)
    r = partition.robot_radius

    # Use boundary margin = r so all PRM points are at least r from cell
    # edges, preventing cross-cell robot collisions.  Fall back to no
    # margin for very narrow cells where inset sampling would fail.
    margin = r
    _test_cfg = _sample_joint_config(poly, 1, r, rng, boundary_margin=margin,
                                     max_tries_per_robot=50)
    if _test_cfg is None:
        margin = 0.0  # cell too narrow for inset sampling

    graph = nx.Graph()
    samples: List[JointConfig] = []

    # Random interior samples
    for _ in range(num_samples):
        cfg = _sample_joint_config(poly, k, r, rng, boundary_margin=margin)
        if cfg is None:
            continue
        if cfg not in graph:
            graph.add_node(cfg)
            samples.append(cfg)

    # Boundary-port anchor configs
    # Port midpoints lie on the cell boundary.  Nudge them r toward the
    # centroid so the pinned position respects the boundary margin and
    # prevents cross-cell collisions at ports.
    cx = sum(v.x().to_double() for v in poly.vertices()) / poly.size()
    cy = sum(v.y().to_double() for v in poly.vertices()) / poly.size()

    port_nodes: Dict[int, JointConfig] = {}
    for boundary_id, (px, py) in boundary_ports.items():
        # Nudge toward centroid by r (or as close as possible).  The
        # anchor must be at least r from the boundary edge containing
        # the port; otherwise two robots at adjacent ports of the same
        # boundary would be < 2r apart.
        dx, dy = cx - px, cy - py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-12:
            if margin > 0:
                # Use the edge inset directly: project the port r
                # along the inward normal (which here is the centroid
                # direction for a port on a single edge).
                nudge = min(margin, max(dist - 1e-6, 0.0))
            else:
                nudge = 1e-6
            npx = px + (nudge / dist) * dx
            npy = py + (nudge / dist) * dy
        else:
            npx, npy = px, py
        cfg = _sample_joint_config(
            poly, k, r, rng, pinned=[(npx, npy)],
            boundary_margin=margin,
        )
        if cfg is None:
            # Retry without margin
            cfg = _sample_joint_config(
                poly, k, r, rng, pinned=[(npx, npy)],
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


# ---------------------------------------------------------------------------
# Ad-hoc per-(cell, timestep) roadmap
# ---------------------------------------------------------------------------

def build_adhoc_roadmap(
    partition: Partition,
    entry_positions: List[Tuple[float, float]],
    exit_positions: List[Tuple[float, float]],
    pinned_positions: List[Tuple[float, float]],
    robot_radius: float,
    num_samples: int = 15,
    k_nearest: int = 6,
    steer_steps: int = DEFAULT_STEER_STEPS,
    rng: Optional[random.Random] = None,
    checker=None,
) -> Optional[Tuple[nx.Graph, JointConfig, JointConfig]]:
    """Build a single-timestep joint PRM for one cell.

    Used by the staged solver at path-realisation time: for each
    ``(cell, timestep)`` with active robots the solver builds a small
    joint PRM whose joint *entry* and joint *exit* configurations are
    inserted as explicit nodes, so there is no "nearest neighbour"
    approximation at the endpoints.

    Slot layout of every node in the returned graph::

        (transit_0, …, transit_{n-1}, holder_0, …, holder_{m-1})

    where ``n = len(entry_positions) = len(exit_positions)`` and
    ``m = len(pinned_positions)``. Holder slots are pinned: every node
    has its holder slots at exactly ``pinned_positions``, so holders do
    not move along any path in the graph. Transit slots vary.

    Args
    ----
    partition : the cell to plan inside.
    entry_positions : current positions of the transit robots.
    exit_positions : where each transit robot wants to end up (same
        order as ``entry_positions``).
    pinned_positions : positions of holding robots inside the cell.
        Pinned to their current location at every node.
    robot_radius : disc robot radius ``r``.
    num_samples : number of random interior configurations to draw
        (in addition to the explicit entry/exit nodes).
    k_nearest : k-NN connection degree for the sampled graph.
    steer_steps : discretisation of the swept-volume pairwise check.
    rng : optional RNG for reproducibility.

    Returns
    -------
    ``(graph, entry_cfg, exit_cfg)`` on success — the caller runs
    ``nx.shortest_path(graph, entry_cfg, exit_cfg, weight='weight')``.
    Returns ``None`` if the entry or exit joint configuration is
    itself infeasible (pairwise 2r separation or cell containment
    violated), which the caller should treat as a hard failure — no
    silent fallback.
    """
    if rng is None:
        rng = random.Random()

    poly = partition.polygon
    r = robot_radius
    min_sq = (2.0 * r) ** 2
    n_transit = len(entry_positions)
    n_pinned = len(pinned_positions)
    k = n_transit + n_pinned

    if len(exit_positions) != n_transit:
        return None
    if k == 0:
        return None

    entry_cfg: JointConfig = tuple(
        list(entry_positions) + list(pinned_positions)
    )
    exit_cfg: JointConfig = tuple(
        list(exit_positions) + list(pinned_positions)
    )

    # Validate entry/exit: all inside cell + pairwise 2r separation.
    for cfg in (entry_cfg, exit_cfg):
        if not _pairwise_separated(list(cfg), min_sq):
            return None
        for x, y in cfg:
            if not _point_inside_polygon(poly, x, y):
                return None

    # Narrow-cell margin fallback: try r-inset sampling first, fall
    # back to 0 if even a single-robot r-inset probe fails.
    margin = r
    probe = _sample_joint_config(
        poly, 1, r, rng, boundary_margin=margin, max_tries_per_robot=50,
        checker=checker,
    )
    if probe is None:
        margin = 0.0

    graph = nx.Graph()
    samples: List[JointConfig] = []

    graph.add_node(entry_cfg)
    samples.append(entry_cfg)
    if exit_cfg != entry_cfg:
        graph.add_node(exit_cfg)
        samples.append(exit_cfg)

    # Random interior samples with holders pinned.
    pinned_arg = list(pinned_positions) if n_pinned > 0 else None
    for _ in range(num_samples):
        raw = _sample_joint_config(
            poly, k, r, rng,
            pinned=pinned_arg,
            boundary_margin=margin,
            checker=checker,
        )
        if raw is None:
            continue
        # ``_sample_joint_config`` returns pinned positions first, then
        # the newly sampled ones. Reorder to (transits…, holders…) so
        # slot indices line up with ``entry_cfg`` and ``exit_cfg``.
        if n_pinned > 0:
            cfg = tuple(list(raw[n_pinned:]) + list(raw[:n_pinned]))
        else:
            cfg = raw
        if cfg in graph:
            continue
        graph.add_node(cfg)
        samples.append(cfg)

    # k-NN connection with swept-volume pairwise check.
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
            if _steer_collision_free(
                ci, cj, r, steer_steps, poly=poly, checker=checker,
            ):
                graph.add_edge(ci, cj, weight=distance)

    return graph, entry_cfg, exit_cfg
