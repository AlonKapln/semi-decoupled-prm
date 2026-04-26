import math
import random
from typing import List, Optional, Sequence, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Pol2, Point_2, Segment_2

from partition import Partition

# Joint config: tuple of (x, y) per robot slot.
# Order: (transit_0, ..., transit_{n-1}, holder_0, ..., holder_{m-1}).
JointConfig = Tuple[Tuple[float, float], ...]

_SAMPLE_TRIES = 200


def _polygon_bbox(poly: Pol2.Polygon_2) -> Tuple[float, float, float, float]:
    """(min_x, max_x, min_y, max_y) of the polygon's vertices."""
    xs = [v.x().to_double() for v in poly.vertices()]
    ys = [v.y().to_double() for v in poly.vertices()]
    return min(xs), max(xs), min(ys), max(ys)


def _point_inside_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
    """True iff (x, y) is inside or on the polygon boundary."""
    side = poly.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _pairwise_separated(
        points: Sequence[Tuple[float, float]], min_dist_sq: float,
) -> bool:
    """True iff every pair of points is at least sqrt(min_dist_sq) apart."""
    for i in range(len(points)):
        xi, yi = points[i]
        for j in range(i + 1, len(points)):
            dx = xi - points[j][0]
            dy = yi - points[j][1]
            if dx * dx + dy * dy < min_dist_sq:
                return False
    return True


def _is_valid_config(
        cfg: JointConfig,
        poly: Pol2.Polygon_2,
        min_dist_sq: float,
        checker=None,
) -> bool:
    """Pairwise 2r-separated, every robot inside poly, every robot passes
    the obstacle checker."""
    if not _pairwise_separated(cfg, min_dist_sq):
        return False
    for x, y in cfg:
        if not _point_inside_polygon(poly, x, y):
            return False
        if checker is not None and not checker.is_point_valid(
                Point_2(FT(x), FT(y))
        ):
            return False
    return True


def _steer_pairwise_min_ok(
        c1: JointConfig, c2: JointConfig, min_dist_sq: float,
) -> bool:
    """Closed-form min of d(t)^2 over t in [0, 1] for every robot pair on
    the straight-line joint steer c1 -> c2.

    :param c1: start joint config.
    :param c2: end joint config.
    :param min_dist_sq: (2r)^2 squared separation threshold.
    :return: True iff no pair gets closer than 2r anywhere on the steer.
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


def _config_distance(c1: JointConfig, c2: JointConfig) -> float:
    """Max per-robot displacement (L-infinity in robot space)."""
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(c1, c2))


def _steer_collision_free(
        c1: JointConfig,
        c2: JointConfig,
        robot_radius: float,
        checker=None,
) -> bool:
    """Pairwise 2r separation + per-robot obstacle check on the straight
    steer c1 -> c2. No cell-containment check: every cell concavity comes
    from obstacle carving, so leaving the cell means hitting an obstacle.

    :param c1: start joint config.
    :param c2: end joint config.
    :param robot_radius: disc radius r.
    :param checker: ObjectCollisionDetection or None.
    :return: True iff the steer is valid.
    """
    min_sq = (2.0 * robot_radius) ** 2
    if not _steer_pairwise_min_ok(c1, c2, min_sq):
        return False
    if checker is not None:
        for i in range(len(c1)):
            if c1[i] == c2[i]:
                continue  # zero-length segment: pinned holder
            seg = Segment_2(
                Point_2(FT(c1[i][0]), FT(c1[i][1])),
                Point_2(FT(c2[i][0]), FT(c2[i][1])),
            )
            if not checker.is_edge_valid(seg):
                return False
    return True


def _sample_joint_config(
        poly: Pol2.Polygon_2,
        n_transit: int,
        holders: List[Tuple[float, float]],
        robot_radius: float,
        rng: random.Random,
        checker=None,
) -> Optional[JointConfig]:
    """Place n_transit robots inside poly, each 2r-separated from
    previously placed robots and from holders.

    :param poly: cell polygon.
    :param n_transit: number of robots to place.
    :param holders: fixed positions kept as obstacles.
    :param robot_radius: disc radius r.
    :param rng: RNG for rejection sampling.
    :param checker: arc-exact obstacle checker, or None.
    :return: tuple (transits..., holders...) on success, None if the
        per-robot _SAMPLE_TRIES budget is exhausted.
    """
    min_sq = (2.0 * robot_radius) ** 2
    if not _pairwise_separated(holders, min_sq):
        return None

    minx, maxx, miny, maxy = _polygon_bbox(poly)
    transits: List[Tuple[float, float]] = []

    while len(transits) < n_transit:
        placed = False
        for _ in range(_SAMPLE_TRIES):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if not _point_inside_polygon(poly, x, y):
                continue
            if not all(
                    (x - tx) ** 2 + (y - ty) ** 2 >= min_sq
                    for tx, ty in transits
            ):
                continue
            if not all(
                    (x - hx) ** 2 + (y - hy) ** 2 >= min_sq
                    for hx, hy in holders
            ):
                continue
            if checker is not None and not checker.is_point_valid(
                    Point_2(FT(x), FT(y))
            ):
                continue
            transits.append((x, y))
            placed = True
            break
        if not placed:
            return None
    return tuple(transits) + tuple(holders)


def build_adhoc_roadmap(
        partition: Partition,
        entry_cfg: JointConfig,
        exit_cfg: JointConfig,
        n_transit: int,
        robot_radius: float,
        num_samples: int = 15,
        k_nearest: int = 6,
        rng: Optional[random.Random] = None,
        checker=None,
) -> Optional[nx.Graph]:
    """Joint k-robot PRM for one cell at one timestep.

    Anchors entry_cfg and exit_cfg as explicit nodes, draws random
    interior samples with holders pinned, then a bridge-burst around each
    anchor's transit slots. Connects via k-NN plus a brute-force pass
    from each anchor.

    :param partition: cell.
    :param entry_cfg: joint config at the start of the timestep.
    :param exit_cfg: joint config at the end. Holder slots
        (entry_cfg[n_transit:]) must equal exit_cfg[n_transit:].
    :param n_transit: number of transit slots.
    :param robot_radius: disc radius r.
    :param num_samples: random interior samples to draw.
    :param k_nearest: k-NN connection degree.
    :param rng: RNG.
    :param checker: arc-exact obstacle checker.
    :return: roadmap graph, or None if either endpoint is invalid.
    """
    if rng is None:
        rng = random.Random()

    poly = partition.polygon
    r = robot_radius
    min_sq = (2.0 * r) ** 2
    pinned_positions: List[Tuple[float, float]] = list(entry_cfg[n_transit:])

    for cfg in (entry_cfg, exit_cfg):
        if not _is_valid_config(cfg, poly, min_sq, checker=None):
            return None

    graph = nx.Graph()
    samples: List[JointConfig] = []

    anchors = [entry_cfg] if exit_cfg == entry_cfg else [entry_cfg, exit_cfg]
    for a in anchors:
        graph.add_node(a)
        samples.append(a)

    for _ in range(num_samples):
        cfg = _sample_joint_config(
            poly, n_transit, pinned_positions,
            r, rng, checker=checker,
        )
        if cfg is None or cfg in graph:
            continue
        graph.add_node(cfg)
        samples.append(cfg)

    # Bridge-burst: perturb one transit slot at a time around each anchor
    # so k-NN can thread the anchor through narrow strips.
    bridge_burst = max(10, num_samples // 2)
    bridge_radius = 3.0 * r
    for anchor in anchors:
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
                if not _pairwise_separated(cfg, min_sq):
                    continue
                if cfg in graph:
                    continue
                graph.add_node(cfg)
                samples.append(cfg)

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
            if _steer_collision_free(ci, cj, r, checker=checker):
                graph.add_edge(ci, cj, weight=distance)

    # Anchor brute-force: cheap, recovers paths k-NN misses at corners.
    for anchor in anchors:
        for other in samples:
            if other is anchor or graph.has_edge(anchor, other):
                continue
            if _steer_collision_free(anchor, other, r, checker=checker):
                graph.add_edge(
                    anchor, other, weight=_config_distance(anchor, other),
                )

    return graph


def plan_adhoc(
        partition: Partition,
        transit_entries: List[Tuple[float, float]],
        transit_exits: List[Tuple[float, float]],
        pinned: List[Tuple[float, float]],
        robot_radius: float,
        num_samples: int,
        k_nearest: int = 6,
        rng: Optional[random.Random] = None,
        checker=None,
) -> Optional[List[JointConfig]]:
    """Build an ad-hoc roadmap with sample escalation (1x/2x/4x/8x) and
    return the shortest joint-config path entry -> exit.

    :param partition: cell.
    :param transit_entries: per-transit-robot start positions.
    :param transit_exits: per-transit-robot end positions.
    :param pinned: holder positions (kept as obstacles).
    :param robot_radius: disc radius r.
    :param num_samples: base sample count for the first attempt.
    :param k_nearest: PRM k-NN degree.
    :param rng: RNG, reused across escalations so successive draws differ.
    :param checker: arc-exact obstacle checker.
    :return: joint-config sequence, or None if no attempt succeeds.
    """
    if rng is None:
        rng = random.Random()

    n_transit = len(transit_entries)
    if len(transit_exits) != n_transit or n_transit + len(pinned) == 0:
        return None

    entry_cfg: JointConfig = tuple(transit_entries) + tuple(pinned)
    exit_cfg: JointConfig = tuple(transit_exits) + tuple(pinned)

    if entry_cfg == exit_cfg:
        return [entry_cfg]  # nobody moves this timestep

    for nsamples in (num_samples, num_samples * 2, num_samples * 4, num_samples * 8):
        graph = build_adhoc_roadmap(
            partition,
            entry_cfg=entry_cfg,
            exit_cfg=exit_cfg,
            n_transit=n_transit,
            robot_radius=robot_radius,
            num_samples=nsamples,
            k_nearest=k_nearest,
            rng=rng,
            checker=checker,
        )
        if graph is None:
            continue
        try:
            return nx.shortest_path(graph, entry_cfg, exit_cfg, weight="weight")
        except nx.NetworkXNoPath:
            continue
    return None
