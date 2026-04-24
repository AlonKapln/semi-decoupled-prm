"""Ad-hoc per-(cell, timestep) joint k-robot PRM.

A joint configuration is valid iff every pair is at least 2r apart,
every robot lies inside the cell polygon, and every robot passes the
scene-level collision checker. Straight-line joint steers are vetted
by a closed-form pairwise-separation minimum plus a swept containment
check.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side

from discopygal.bindings import FT, Pol2, Point_2, Segment_2

from partition import Partition


# A joint configuration is a tuple of (x, y) tuples, one per robot slot.
JointConfig = Tuple[Tuple[float, float], ...]

DEFAULT_STEER_STEPS = 10


@dataclass
class AdhocResult:
    """graph is None on failure; reason is one of entry_infeasible,
    exit_infeasible (endpoint bad, resampling won't help), no_path, or
    sampling_failed (more samples may help)."""

    graph: Optional[nx.Graph]
    entry_cfg: JointConfig
    exit_cfg: JointConfig
    interior_samples: List[JointConfig] = field(default_factory=list)
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Cell geometry helpers
# ---------------------------------------------------------------------------

def _polygon_bbox(poly: Pol2.Polygon_2) -> Tuple[float, float, float, float]:
    xs = [v.x().to_double() for v in poly.vertices()]
    ys = [v.y().to_double() for v in poly.vertices()]
    return min(xs), max(xs), min(ys), max(ys)


def _point_inside_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
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
    """Closed-form minimum of d(t)^2 over t in [0, 1] for every robot
    pair along a straight-line joint steer."""
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
                # Parallel or zero relative velocity: constant distance.
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
    """Signed distance to the nearest polygon edge (positive = inside, CCW)."""
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
        d = (ex * (y - ay) - ey * (x - ax)) / length
        if d < min_d:
            min_d = d
    return min_d


def _config_distance(c1: JointConfig, c2: JointConfig) -> float:
    # Joint motion is rate-limited by the slowest robot, so the max
    # single-robot displacement is the natural steer cost.
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(c1, c2))


def _steer_collision_free(
    c1: JointConfig,
    c2: JointConfig,
    robot_radius: float,
    steps: int = DEFAULT_STEER_STEPS,
    poly: Optional[Pol2.Polygon_2] = None,
    checker=None,
) -> bool:
    """Straight-line joint steer: exact pairwise separation, optional
    polygon containment (non-convex grid cells), and scene-level
    swept-disc validation via checker (arc-vs-chord gap)."""
    min_sq = (2.0 * robot_radius) ** 2
    k = len(c1)
    if not _steer_pairwise_min_ok(c1, c2, min_sq):
        return False
    # The discretised sweep is an approximation for non-convex cells;
    # the checker below is authoritative for arc-adjacent regions.
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
            # Skip degenerate zero-length segments (pinned holders) to
            # avoid CGAL degenerate-segment asserts.
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
    """Place k robots incrementally: inside polygon, at least boundary_margin
    from edges, pairwise at least 2r apart. Pinned slots are fixed."""
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
            # Arc-exact check for non-convex cells: the polygon chord-
            # approximates inflated-obstacle boundaries.
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
    """Single-timestep joint PRM for one cell.

    Slot layout: (transit_0, ..., transit_{n-1}, holder_0, ..., holder_{m-1}).
    Entry/exit configs are inserted as explicit nodes; holders are pinned
    at every node. On failure the returned graph is None and reason names
    the failure mode.
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

    # Validate entry/exit: inside cell + pairwise 2r separation.
    for cfg, tag in ((entry_cfg, "entry_infeasible"),
                     (exit_cfg, "exit_infeasible")):
        if not _pairwise_separated(list(cfg), min_sq):
            return AdhocResult(None, entry_cfg, exit_cfg, reason=tag)
        for x, y in cfg:
            if not _point_inside_polygon(poly, x, y):
                return AdhocResult(None, entry_cfg, exit_cfg, reason=tag)

    # Narrow-cell margin fallback: try r-margin sampling first, fall
    # back to 0 if even a single-robot r-margin probe fails.
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

    # Reused samples from the solver's cache; each must be re-validated
    # because density retries and checker state can invalidate a
    # previously-good sample.
    if reuse_samples:
        rounded_pinned = tuple(
            (round(x, 6), round(y, 6)) for x, y in pinned_positions
        )
        for cfg in reuse_samples:
            if len(cfg) != k:
                continue
            # Holder slots must match pinned; round both sides to the
            # same 6dp as the cache key so sub-precision drift does not
            # silently reject every reused sample.
            if n_pinned > 0 and tuple(
                (round(x, 6), round(y, 6)) for x, y in cfg[n_transit:]
            ) != rounded_pinned:
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
        # Reorder sampler output (pinned, transits) -> (transits, holders).
        if n_pinned > 0:
            cfg = tuple(list(raw[n_pinned:]) + list(raw[:n_pinned]))
        else:
            cfg = raw
        if cfg in graph:
            continue
        graph.add_node(cfg)
        samples.append(cfg)
        interior_samples.append(cfg)

    # Anchor-local bridge samples: seed a burst around every transit
    # slot of every anchor so k-NN can thread the anchor through a
    # local chain when random sampling misses the narrow strip.
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

    # k-NN connections with swept-volume pairwise check.
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

    # Anchor brute-force: try every straight-line steer from each anchor.
    # Cheap at this sample size and recovers paths k-NN misses at corners.
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
