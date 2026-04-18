"""Ad-hoc per-(cell, timestep) joint k-robot PRM.

Step 4 / step 7 of ``plan.tex``. At path realisation time the staged
solver builds one small joint roadmap per cell per MCF timestep via
:func:`build_adhoc_roadmap`. A *joint* configuration places up to ``k``
robots simultaneously in the cell; the PRM finds a collision-free joint
steer from an explicit entry configuration to an explicit exit
configuration.

Geometry
--------
A joint configuration ``c = (p_1, …, p_k)`` is **valid** when

    ∀ i ≠ j :  ‖p_i − p_j‖ ≥ 2r          (pairwise non-overlap)
    ∀ i     :  p_i ∈ cell                (polygon containment)
    ∀ i     :  p_i ∉ true_obstacles      (checker, when supplied)

Two valid joint configurations are **connected** by a straight-line
joint-space steer iff every interpolated configuration is also valid.
For convex cells this collapses to a pairwise 2r check (solved exactly
in closed form by ``_steer_pairwise_min_ok``); for non-convex cells
from grid-only partitioning we additionally sweep polygon containment
and check each robot's segment against the scene collision checker.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side

from discopygal.bindings import FT, Pol2, Point_2, Segment_2

from partition import Partition


# A joint configuration is a tuple of (x, y) tuples — one per robot slot.
JointConfig = Tuple[Tuple[float, float], ...]

DEFAULT_STEER_STEPS = 10


@dataclass
class AdhocResult:
    """Return value of :func:`build_adhoc_roadmap`.

    ``graph`` is ``None`` on failure; in that case ``reason`` names the
    failure mode. The caller (``StagedSolver._plan_adhoc``) uses ``reason``
    to decide whether escalating ``num_samples`` can help (``no_path`` and
    ``sampling_failed`` → yes; ``entry_infeasible`` / ``exit_infeasible``
    → no, the problem is the fixed endpoints).

    ``interior_samples`` is the list of pure-random interior configs that
    ended up accepted this build (i.e. not entry/exit, not bridge-burst).
    The solver caches this per ``(cell_id, n_transit, pinned)`` and feeds
    it back as ``reuse_samples`` on the next call with the same
    configuration — saves the cost of re-drawing valid joint configs for
    repeat visits to the same cell.
    """

    graph: Optional[nx.Graph]
    entry_cfg: JointConfig
    exit_cfg: JointConfig
    interior_samples: List[JointConfig] = field(default_factory=list)
    reason: Optional[str] = None
    # Reason values: None (success), "entry_infeasible",
    # "exit_infeasible", "no_path", "sampling_failed".


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


def _steer_pairwise_min_ok(
    c1: "JointConfig", c2: "JointConfig", min_dist_sq: float,
) -> bool:
    """Exact pairwise separation over a straight-line joint steer.

    For each robot pair ``(i, j)`` we have two linear trajectories
    ``p_i(t) = c1[i] + t * (c2[i] - c1[i])`` and similarly for ``p_j``.
    Their squared distance ``d²(t) = |p_i(t) - p_j(t)|²`` is a quadratic
    in ``t``, minimised at a closed-form ``t*`` (clamped to ``[0, 1]``).
    We compute that minimum exactly for every pair — no discretisation
    miss, no false accepts from coarse step counts.
    """
    k = len(c1)
    for i in range(k):
        a1x, a1y = c1[i]
        b1x, b1y = c2[i]
        dx1 = b1x - a1x
        dy1 = b1y - a1y
        for j in range(i + 1, k):
            a2x, a2y = c1[j]
            b2x, b2y = c2[j]
            dx2 = b2x - a2x
            dy2 = b2y - a2y
            # diff(t) = (c1[i] - c1[j]) + t * ((c2[i]-c1[i]) - (c2[j]-c1[j]))
            px = a1x - a2x
            py = a1y - a2y
            rx = dx1 - dx2
            ry = dy1 - dy2
            rr = rx * rx + ry * ry
            if rr <= 1e-18:
                # Parallel / zero relative velocity — constant distance.
                if px * px + py * py < min_dist_sq:
                    return False
                continue
            t_star = -(px * rx + py * ry) / rr
            if t_star < 0.0:
                t_star = 0.0
            elif t_star > 1.0:
                t_star = 1.0
            ex = px + t_star * rx
            ey = py + t_star * ry
            if ex * ex + ey * ey < min_dist_sq:
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
    # Exact pairwise separation along the straight-line joint steer.
    if not _steer_pairwise_min_ok(c1, c2, min_sq):
        return False
    # Polygon containment is only approximate here; the discretised
    # sweep still catches the typical cases and the obstacle checker
    # below is authoritative for arcs.
    if poly is not None:
        for s in range(steps + 1):
            t = s / steps
            for i in range(k):
                x = c1[i][0] + t * (c2[i][0] - c1[i][0])
                y = c1[i][1] + t * (c2[i][1] - c1[i][1])
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
    reuse_samples: Optional[List[JointConfig]] = None,
) -> AdhocResult:
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
    An :class:`AdhocResult`. On success ``result.graph`` is a populated
    networkx graph and ``result.reason is None``; the caller runs
    ``nx.shortest_path(result.graph, result.entry_cfg, result.exit_cfg,
    weight='weight')``. On failure ``result.graph is None`` and
    ``result.reason`` names the failure mode so the caller can decide
    whether more samples would help.
    """
    if rng is None:
        rng = random.Random()

    poly = partition.polygon
    r = robot_radius
    min_sq = (2.0 * r) ** 2
    n_transit = len(entry_positions)
    n_pinned = len(pinned_positions)
    k = n_transit + n_pinned

    entry_cfg: JointConfig = tuple(
        list(entry_positions) + list(pinned_positions)
    )
    exit_cfg: JointConfig = tuple(
        list(exit_positions) + list(pinned_positions)
    )

    if len(exit_positions) != n_transit or k == 0:
        return AdhocResult(None, entry_cfg, exit_cfg, reason="entry_infeasible")

    # Validate entry/exit: all inside cell + pairwise 2r separation.
    for cfg, tag in ((entry_cfg, "entry_infeasible"),
                     (exit_cfg, "exit_infeasible")):
        if not _pairwise_separated(list(cfg), min_sq):
            return AdhocResult(None, entry_cfg, exit_cfg, reason=tag)
        for x, y in cfg:
            if not _point_inside_polygon(poly, x, y):
                return AdhocResult(None, entry_cfg, exit_cfg, reason=tag)

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
    interior_samples: List[JointConfig] = []

    graph.add_node(entry_cfg)
    samples.append(entry_cfg)
    if exit_cfg != entry_cfg:
        graph.add_node(exit_cfg)
        samples.append(exit_cfg)

    # Reused samples from the solver's cache (same cell, same n_transit,
    # same pinned set). Re-validate each one — cell density may have been
    # lowered by a Level B retry or the checker state may differ, so a
    # previously-valid sample is not automatically valid now.
    if reuse_samples:
        for cfg in reuse_samples:
            if len(cfg) != k:
                continue
            # Holder slots must still match current pinned positions.
            if n_pinned > 0 and tuple(cfg[n_transit:]) != tuple(pinned_positions):
                continue
            if cfg in graph:
                continue
            if not _pairwise_separated(list(cfg), min_sq):
                continue
            bad = False
            for x, y in cfg:
                if not _point_inside_polygon(poly, x, y):
                    bad = True
                    break
                if checker is not None and not checker.is_point_valid(
                    Point_2(FT(x), FT(y))
                ):
                    bad = True
                    break
            if bad:
                continue
            graph.add_node(cfg)
            samples.append(cfg)
            interior_samples.append(cfg)

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
        interior_samples.append(cfg)

    # Anchor-local bridge samples. Validated ports already sit in
    # regions with a 2r escape, but the main random sampler still
    # has a hard time hitting the narrow strip near the anchor when
    # the cell is large relative to that strip. Seed a burst of extra
    # samples in a small window around every transit slot of every
    # anchor so k-NN can thread the anchor through a local chain.
    bridge_burst = max(10, num_samples // 2)
    bridge_radius = max(3.0 * r, 0.4)
    anchor_list = [entry_cfg] if exit_cfg == entry_cfg else [entry_cfg, exit_cfg]
    for anchor in anchor_list:
        for slot in range(n_transit):
            ax, ay = anchor[slot]
            for _ in range(bridge_burst):
                bx = ax + rng.uniform(-bridge_radius, bridge_radius)
                by = ay + rng.uniform(-bridge_radius, bridge_radius)
                if not _point_inside_polygon(poly, bx, by):
                    continue
                if checker is not None and not checker.is_point_valid(
                    Point_2(FT(bx), FT(by))
                ):
                    continue
                trial = list(anchor[:n_transit])
                trial[slot] = (bx, by)
                trial.extend(anchor[n_transit:])
                cfg = tuple(trial)
                if not _pairwise_separated(list(cfg), min_sq):
                    continue
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

    # Anchor brute-force pass: try every (entry, other) and (exit, other)
    # straight-line steer. ``num_samples`` is small, so the O(N) per
    # anchor is cheap and it recovers paths that k-NN misses when the
    # anchor sits near an awkward corner.
    anchors = [entry_cfg]
    if exit_cfg != entry_cfg:
        anchors.append(exit_cfg)
    for anchor in anchors:
        for other in samples:
            if other is anchor or graph.has_edge(anchor, other):
                continue
            if _steer_collision_free(
                anchor, other, r, steer_steps, poly=poly, checker=checker,
            ):
                graph.add_edge(
                    anchor, other, weight=_config_distance(anchor, other),
                )

    # If only the explicit entry/exit nodes made it into the graph (every
    # random sample was rejected), surface that as its own reason so the
    # caller can escalate num_samples rather than lower cell density.
    if len(samples) <= (2 if entry_cfg != exit_cfg else 1):
        return AdhocResult(
            graph, entry_cfg, exit_cfg,
            interior_samples=interior_samples,
            reason="sampling_failed",
        )

    return AdhocResult(
        graph, entry_cfg, exit_cfg,
        interior_samples=interior_samples,
        reason=None,
    )
