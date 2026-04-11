"""High-level cell-adjacency graph for multi-commodity flow.

Step 5 of ``plan.tex``.  Given the trapezoidal partitions from step 2 and
robot start/goal positions, this module builds a graph whose:

- **Nodes** are boundary ports (midpoints of shared edges between cells)
  plus robot start and goal positions.
- **Edges** represent traversal through a cell, connecting pairs of
  incident nodes.  Edge attributes include the cell index and capacity.

Two graph topologies are supported (selectable via the ``topology``
parameter) so their behaviour can be compared:

``"pairwise"``
    Every pair of nodes incident to the same cell is connected by a direct
    edge.  Simple, O(m²) edges per cell where m is the number of incident
    nodes (typically 3–5 for trapezoids).  Each cell traversal is one hop.

``"star"``
    Each cell gets a virtual **hub** node at its centroid.  Incident nodes
    connect to the hub instead of to each other.  O(m) edges per cell, and
    the hub naturally enforces per-cell capacity, but each cell traversal
    takes **two** hops (enter hub then leave hub), which inflates the
    time-expanded ILP.

Adjacency discovery
-------------------
Two trapezoidal cells are adjacent when their ``Pol2.Polygon_2`` outlines
share a complete edge (a pair of consecutive vertices).  After vertical
decomposition every arrangement edge is maximally split, so shared edges
appear as identical vertex pairs in both polygons.  We round vertex
coordinates to ``_COORD_PRECISION`` decimal places to absorb any float
noise from the ``a0()`` rational projection.

Boundary-port positions
-----------------------
Each shared edge produces **one or more** port nodes, placed along
subsegments where an ``r``-inward inset is valid on both sides. Long
edges (grid-decomposition boundaries between large open cells) get
multiple ports spaced by ``_MIN_PORT_SPACING_R * robot_radius`` so MCF
sees parallel capacity channels and doesn't bottleneck the whole edge
on a single transit slot. Edges with no usable subsegment are dropped.

The ``cell_boundary_ports`` output maps each cell index to a ``{port_id:
inset_position_inside_that_cell}`` dict: the inset is precomputed at HLG
build time so path realisation can look it up directly without rerunning
the geometry search.

Time-horizon estimation
-----------------------
``estimate_time_horizon`` counts the maximum unweighted shortest-path
length (in hops) from any robot's start node to its goal node and
multiplies by a congestion factor, matching the heuristic in
``multi_robot_flow_solver.py:224-236``.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2, Pol2

from partition import Partition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Round vertex coordinates to this many decimal places when comparing edges
# across cells.  Vertices come from _face_to_polygon's a0() projection so
# they should be bit-identical; rounding guards against float noise.
_COORD_PRECISION = 10

# Shift port insets along the shared-edge tangent (in opposite
# directions per side) so opposing insets are not directly perpendicular
# across from each other. Without this, both sides inset by exactly r
# land 2r apart and discopygal's swept-disc collision check reports
# tangent contact between adjacent-cell robots as a collision. The
# tangential offset adds lateral separation without demanding more
# perpendicular inset than the cell can support.
_PORT_TANGENT_OFFSET = 0.2

# Minimum along-edge spacing between adjacent ports on the same shared
# edge, in units of robot radius. Two robots simultaneously transiting
# adjacent ports sit this far apart along the wall; they also each need
# room to swing inward by r toward their in-cell insets. Empirically,
# spacings below ~8r make the ad-hoc joint PRM struggle (two transit
# robots sharing a cell via neighbouring ports can't find valid
# intermediate configs), while spacings above ~16r leave MCF
# under-utilising long walls. 10r is the sweet spot.
_MIN_PORT_SPACING_R = 10.0

# Margin (in units of robot radius) kept between any port and the
# endpoints of its shared edge. Two distinct shared edges of the same
# cell meet at a vertex; a port placed near that vertex on edge A ends
# up arbitrarily close to a port placed near the same vertex on edge B,
# which breaks the joint PRM's 2r pairwise separation. Trimming the
# valid region by this margin at both ends of every edge pushes ports
# away from corners and keeps ports on different shared edges of the
# same cell from landing on top of each other.
_PORT_EDGE_END_MARGIN_R = 3.0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HighLevelGraph:
    """Result of :func:`build_high_level_graph`.

    Attributes
    ----------
    graph : nx.Graph
        Nodes are strings (``"port_0"``, ``"start_1"``, ``"goal_2"``,
        ``"hub_3"``).  Node attributes include ``kind`` (``"port"``,
        ``"start"``, ``"goal"``, ``"hub"``).  Edge attributes:

        - ``cell_id`` (int) — which cell this traversal passes through.
        - ``capacity`` (int) — cell density (max simultaneous robots).
        - ``cost`` (float) — Euclidean distance between endpoints.

    node_positions : dict[str, (float, float)]
        Geometric position for every node.
    cell_boundary_ports : dict[int, dict[int, (float, float)]]
        For each cell index, ``{port_id: (x, y)}`` where ``(x, y)`` is the
        precomputed ``r``-inset of the port *inside that cell* (not the
        raw edge midpoint). Used directly by path realisation as the
        joint-PRM entry/exit position for a robot transiting through the
        cell via that port.
    cell_incident_nodes : dict[int, list[str]]
        For each cell index, the list of node IDs incident to that cell
        (ports, starts, goals — and hubs in star topology).
    start_cells : dict[int, int]
        Maps robot index → cell index containing the robot's start.
    goal_cells : dict[int, int]
        Maps robot index → cell index containing the robot's goal.
    topology : str
        ``"pairwise"`` or ``"star"``.
    """

    graph: nx.Graph
    node_positions: Dict[str, Tuple[float, float]]
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]]
    cell_incident_nodes: Dict[int, List[str]]
    start_cells: Dict[int, int]
    goal_cells: Dict[int, int]
    topology: str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_in_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
    """Closed (boundary-inclusive) containment test."""
    side = poly.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _polygon_centroid(poly: Pol2.Polygon_2) -> Tuple[float, float]:
    """ Average over X's and Y's of vertices in a polygon will give us the centeroid of the polygon. """
    xs = [v.x().to_double() for v in poly.vertices()]
    ys = [v.y().to_double() for v in poly.vertices()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Adjacency discovery
# ---------------------------------------------------------------------------

def _find_shared_edges(
        partitions: List[Partition],
) -> List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]:
    """Discover edges shared between partition polygons.

    Returns a list of ``(cell_i, cell_j, v1, v2)`` tuples, one per shared
    edge.  ``cell_i < cell_j`` always.  ``v1`` and ``v2`` are the two
    endpoints of the shared edge (order is canonical — sorted by
    coordinate — so both callers see the same pair).
    """
    # For each cell build a set of canonical edge keys: sorted vertex pairs.
    cell_edge_sets: List[set] = []
    for p in partitions:
        verts = [
            (
                round(v.x().to_double(), _COORD_PRECISION),
                round(v.y().to_double(), _COORD_PRECISION),
            )
            for v in p.polygon.vertices()
        ]
        edges = set()
        n = len(verts)
        for i in range(n):
            e = tuple(sorted([verts[i], verts[(i + 1) % n]]))
            edges.add(e)
        cell_edge_sets.append(edges)

    boundaries: List[
        Tuple[int, int, Tuple[float, float], Tuple[float, float]]
    ] = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            shared = cell_edge_sets[i] & cell_edge_sets[j]
            for edge in shared:
                v1, v2 = edge
                boundaries.append((i, j, v1, v2))
    return boundaries


def _edge_inward_normal(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        poly: Pol2.Polygon_2,
) -> Optional[Tuple[float, float]]:
    """Unit vector from the edge midpoint into the cell interior.

    Probes both perpendicular directions and returns whichever one
    lands inside the polygon. Returns ``None`` for degenerate edges or
    when neither probe is inside (e.g. polygon orientation is weird).
    """
    dx, dy = v2[0] - v1[0], v2[1] - v1[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    nx1, ny1 = -dy / length, dx / length
    mx = (v1[0] + v2[0]) / 2.0
    my = (v1[1] + v2[1]) / 2.0
    eps = 1e-6
    if _point_in_polygon(poly, mx + eps * nx1, my + eps * ny1):
        return (nx1, ny1)
    if _point_in_polygon(poly, mx - eps * nx1, my - eps * ny1):
        return (-nx1, -ny1)
    return None


def _max_valid_inset(
        px: float,
        py: float,
        nx_: float,
        ny_: float,
        poly: Pol2.Polygon_2,
        robot_radius: float,
        checker,
) -> Optional[Tuple[float, float]]:
    """Largest valid inset along ``(nx_, ny_)`` up to ``r``, or ``None``.

    Tries a decreasing sequence of fractions of ``r``. Accepts the first
    candidate that is inside ``poly`` and (if a checker is supplied)
    checker-valid. Returns the resulting point, or ``None`` if none work.

    Allowing sub-``r`` insets handles decomposition artifacts — thin
    slivers narrower than ``r`` that a robot physically transits without
    ever needing 2r of clearance on both sides. For dead-end pockets in
    larger cells, the ``_has_2r_escape`` check below catches them
    separately.
    """
    fractions = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.025, 0.01)
    for frac in fractions:
        cx = px + frac * robot_radius * nx_
        cy = py + frac * robot_radius * ny_
        if not _point_in_polygon(poly, cx, cy):
            continue
        if checker is not None and not checker.is_point_valid(
            Point_2(FT(cx), FT(cy)),
        ):
            continue
        return (cx, cy)
    return None


def _has_2r_escape(
        x: float,
        y: float,
        poly: Pol2.Polygon_2,
        robot_radius: float,
        checker,
        n_dirs: int = 8,
        n_steps: int = 5,
) -> bool:
    """True if a 2r-length straight line from ``(x, y)`` stays valid.

    Tests ``n_dirs`` evenly spaced directions. For each, samples
    ``n_steps`` interior points along the segment; if all are inside
    the polygon and checker-valid, the direction is an escape.

    This distinguishes dead-end pockets (no 2r line fits anywhere — e.g.
    the triangle wedged above an inflated obstacle corner) from narrow
    decomposition slivers, whose lengthwise direction easily admits a 2r
    straight line.
    """
    d = 2.0 * robot_radius
    for k in range(n_dirs):
        angle = 2.0 * math.pi * k / n_dirs
        dx = math.cos(angle) * d
        dy = math.sin(angle) * d
        ok = True
        for s in range(1, n_steps + 1):
            t = s / n_steps
            px = x + t * dx
            py = y + t * dy
            if not _point_in_polygon(poly, px, py):
                ok = False
                break
            if checker is not None and not checker.is_point_valid(
                Point_2(FT(px), FT(py)),
            ):
                ok = False
                break
        if ok:
            return True
    return False


def _sample_valid_insets(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        poly_i: Pol2.Polygon_2,
        poly_j: Pol2.Polygon_2,
        robot_radius: float,
        checker,
        samples: int,
) -> List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """For each sample index along the edge, return the tangent-shifted
    (inset_i, inset_j) pair if it is valid, else ``None``.

    Validity requires: per-side ``_max_valid_inset`` succeeds, the tangent-
    shifted variant stays inside both polygons and checker-valid, and both
    insets have a 2r straight-line escape inside their cell (not a
    dead-end pocket).
    """
    ni = _edge_inward_normal(v1, v2, poly_i)
    nj = _edge_inward_normal(v1, v2, poly_j)
    if ni is None or nj is None:
        return [None] * samples

    tlen = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    if tlen > 1e-12:
        tx = (v2[0] - v1[0]) / tlen
        ty = (v2[1] - v1[1]) / tlen
    else:
        tx = ty = 0.0
    off = robot_radius * _PORT_TANGENT_OFFSET

    entries: List[
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    ] = [None] * samples

    for k in range(samples):
        t = k / (samples - 1)
        px = v1[0] + t * (v2[0] - v1[0])
        py = v1[1] + t * (v2[1] - v1[1])
        pi = _max_valid_inset(
            px, py, ni[0], ni[1], poly_i, robot_radius, checker,
        )
        if pi is None:
            continue
        pj = _max_valid_inset(
            px, py, nj[0], nj[1], poly_j, robot_radius, checker,
        )
        if pj is None:
            continue

        # Tangent offset: shift pi and pj in opposite directions along
        # the shared edge so opposing insets are NOT directly across
        # from each other. Without this, both sides inset by exactly r
        # along the perpendicular and sit exactly 2r apart, which
        # discopygal's verify_paths reports as a tangent-disc collision
        # when two robots cross the edge simultaneously.
        if off > 0.0:
            pi_shift = (pi[0] + tx * off, pi[1] + ty * off)
            pj_shift = (pj[0] - tx * off, pj[1] - ty * off)
            if (
                _point_in_polygon(poly_i, pi_shift[0], pi_shift[1])
                and (
                    checker is None
                    or checker.is_point_valid(
                        Point_2(FT(pi_shift[0]), FT(pi_shift[1]))
                    )
                )
                and _point_in_polygon(poly_j, pj_shift[0], pj_shift[1])
                and (
                    checker is None
                    or checker.is_point_valid(
                        Point_2(FT(pj_shift[0]), FT(pj_shift[1]))
                    )
                )
            ):
                pi = pi_shift
                pj = pj_shift

        if not _has_2r_escape(pi[0], pi[1], poly_i, robot_radius, checker):
            continue
        if not _has_2r_escape(pj[0], pj[1], poly_j, robot_radius, checker):
            continue
        entries[k] = (pi, pj)
    return entries


def _contiguous_runs(
        entries: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` (inclusive) index pairs for each maximal run
    of non-``None`` sample indices."""
    runs: List[Tuple[int, int]] = []
    n = len(entries)
    cur = -1
    for k in range(n + 1):
        valid = k < n and entries[k] is not None
        if valid and cur < 0:
            cur = k
        elif not valid and cur >= 0:
            runs.append((cur, k - 1))
            cur = -1
    return runs


def _validated_port_positions(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        poly_i: Pol2.Polygon_2,
        poly_j: Pol2.Polygon_2,
        robot_radius: float,
        checker,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Place **one or more** ports along a shared edge.

    Long edges — common in grid-decomposed scenes where two large open
    cells share a wall many times the robot radius — benefit from having
    multiple ports in parallel: MCF sees each port as an independent
    transit slot, so the per-timestep flow across that wall is no longer
    bottlenecked by a single entry point.

    Algorithm
    ---------
    1. Densely sample the edge; for each sample, compute the tangent-
       shifted ``(inset_i, inset_j)`` pair (``_sample_valid_insets``).
    2. Group into maximal contiguous runs of valid samples.
    3. For each run, compute how many ports fit with a minimum along-
       edge spacing of ``_MIN_PORT_SPACING_R * r``. Distribute them at
       uniform ``t`` within the run and snap each to the nearest valid
       sample.

    Returns a list of ``(midpoint, inset_i, inset_j)`` tuples — possibly
    empty when no point on the edge admits a valid inset on both sides.
    """
    edge_len = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    if edge_len < 1e-12:
        return []

    # Resolution: enough samples that a port spacing of 0.5r is well
    # resolved along the edge. Minimum 21 samples so short edges still
    # get a centre probe.
    samples = max(21, int(edge_len / (0.5 * robot_radius)) + 1)

    entries = _sample_valid_insets(
        v1, v2, poly_i, poly_j, robot_radius, checker, samples,
    )

    # Trim the sampled edge by an end margin so ports never land on top
    # of an edge endpoint. Two different shared edges of the same cell
    # meet at such endpoints, and corner-adjacent ports on the two
    # edges can end up arbitrarily close in cell interior. We use
    # min(margin, 40% of edge length / 2) so that short edges still
    # admit a centre port.
    margin = _PORT_EDGE_END_MARGIN_R * robot_radius
    max_margin = 0.4 * edge_len
    if margin > max_margin:
        margin = max_margin
    t_margin = margin / edge_len if edge_len > 0 else 0.0
    k_margin_start = int(math.ceil(t_margin * (samples - 1)))
    k_margin_end = int((1.0 - t_margin) * (samples - 1))

    min_spacing = _MIN_PORT_SPACING_R * robot_radius
    out: List[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = []

    for (s, e) in _contiguous_runs(entries):
        # Clip the run against the edge-end margin.
        s = max(s, k_margin_start)
        e = min(e, k_margin_end)
        if s > e:
            continue
        t_s = s / (samples - 1)
        t_e = e / (samples - 1)
        run_len = edge_len * (t_e - t_s)

        # Ports fit: n points with (n-1) gaps of >= min_spacing => n-1 <=
        # run_len / min_spacing. Always at least 1.
        n_ports = max(1, 1 + int(run_len // min_spacing))

        for p in range(n_ports):
            if n_ports == 1:
                t_target = (t_s + t_e) / 2.0
            else:
                t_target = t_s + (t_e - t_s) * (p / (n_ports - 1))
            k = round(t_target * (samples - 1))
            k = max(s, min(e, k))
            if entries[k] is None:
                continue
            pair = entries[k]
            assert pair is not None  # satisfied by the check above
            inset_i, inset_j = pair
            t_mid = k / (samples - 1)
            midpoint = (
                v1[0] + t_mid * (v2[0] - v1[0]),
                v1[1] + t_mid * (v2[1] - v1[1]),
            )
            out.append((midpoint, inset_i, inset_j))

    return out


def _find_containing_cell(
        partitions: List[Partition],
        x: float,
        y: float,
) -> Optional[int]:
    """Return the index of the first partition containing ``(x, y)``."""
    for idx, p in enumerate(partitions):
        if _point_in_polygon(p.polygon, x, y):
            return idx
    return None


# ---------------------------------------------------------------------------
# Edge-wiring strategies
# ---------------------------------------------------------------------------

def _add_pairwise_edges(
        G: nx.Graph,
        partitions: List[Partition],
        cell_incident: Dict[int, List[str]],
        node_positions: Dict[str, Tuple[float, float]],
) -> None:
    """Connect every pair of incident nodes within each cell (O(m²))."""
    for ci, nodes in cell_incident.items():
        cap = partitions[ci].density
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                na, nb = nodes[a], nodes[b]
                ax, ay = node_positions[na]
                bx, by = node_positions[nb]
                cost = math.hypot(ax - bx, ay - by)
                G.add_edge(na, nb, cell_id=ci, capacity=cap, cost=cost)


def _add_star_edges(
        G: nx.Graph,
        partitions: List[Partition],
        cell_incident: Dict[int, List[str]],
        node_positions: Dict[str, Tuple[float, float]],
) -> None:
    """Connect each incident node to a virtual hub at the cell centroid."""
    for ci, nodes in cell_incident.items():
        if not nodes:
            continue
        cap = partitions[ci].density
        hub = f"hub_{ci}"
        cx, cy = _polygon_centroid(partitions[ci].polygon)
        G.add_node(hub, kind="hub", cell_id=ci)
        node_positions[hub] = (cx, cy)
        cell_incident[ci].append(hub)
        for n in nodes:
            if n == hub:
                continue
            nx_, ny_ = node_positions[n]
            cost = math.hypot(nx_ - cx, ny_ - cy)
            G.add_edge(n, hub, cell_id=ci, capacity=cap, cost=cost)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_high_level_graph(
        partitions: List[Partition],
        robot_starts: List[Tuple[float, float]],
        robot_goals: List[Tuple[float, float]],
        robot_radius: float,
        topology: str = "pairwise",
        checker=None,
) -> HighLevelGraph:
    """Build the high-level cell graph for MCF routing.

    Args
    ----
    partitions :
        Trapezoidal cells from ``partition_free_space_vertical`` (step 2).
    robot_starts :
        ``[(x, y), …]`` start position for each robot.
    robot_goals :
        ``[(x, y), …]`` goal position for each robot.
    robot_radius :
        Disc robot radius. Used to shrink the cell by ``r`` near each
        shared edge when placing ports: the port is located at a point
        on the edge where an ``r``-inward inset stays inside both
        adjacent cells (so two robots crossing the same shared edge can
        sit ``2r`` apart without collision).
    topology :
        ``"pairwise"`` — O(m²) edges per cell, one hop per traversal.
        ``"star"`` — O(m) edges per cell, two hops per traversal.
    checker :
        Optional ``ObjectCollisionDetection`` for authoritative free-
        space tests against inflated obstacles. When supplied, port
        positions are additionally verified to be checker-valid on
        both sides — essential for non-convex cells from grid-only
        partitioning where the polygon chord-approximates an arc.
    """
    if topology not in ("pairwise", "star"):
        raise ValueError(
            f"topology must be 'pairwise' or 'star', got {topology!r}"
        )

    G = nx.Graph()
    node_positions: Dict[str, Tuple[float, float]] = {}
    cell_incident: Dict[int, List[str]] = {
        i: [] for i in range(len(partitions))
    }
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]] = {
        i: {} for i in range(len(partitions))
    }

    # ---- Boundary port nodes ----
    # For each shared edge, search along the edge for points whose
    # r-inward inset is valid in both neighbouring cells. Long edges
    # (grid decomposition between large open cells) get multiple ports
    # spaced by ``_MIN_PORT_SPACING_R * r`` so MCF sees parallel
    # transit capacity instead of bottlenecking the whole wall on a
    # single entry. Edges with no usable subsegment are skipped.
    #
    # Cross-edge proximity: a single cell can share edges with many
    # neighbours (corner cells in grid decomposition touch 3+ cells),
    # and two ports placed near a shared corner on adjacent shared
    # edges can land arbitrarily close in the cell interior — far
    # below 2r, which breaks the joint PRM's pairwise-separation
    # guarantee. We track already-accepted insets per cell and reject
    # any candidate whose inset in *either* adjacent cell is within
    # ``_MIN_PORT_SPACING_R * r`` of an existing inset in that cell.
    # Precomputed insets go into ``cell_boundary_ports`` so path
    # realisation can look them up without rerunning the search.
    boundaries = _find_shared_edges(partitions)
    port_id = 0
    per_cell_insets: Dict[int, List[Tuple[float, float]]] = {
        i: [] for i in range(len(partitions))
    }
    min_inset_sep_sq = (_MIN_PORT_SPACING_R * robot_radius) ** 2

    def _too_close_to_existing(cell: int, point: Tuple[float, float]) -> bool:
        for (qx, qy) in per_cell_insets[cell]:
            dx = point[0] - qx
            dy = point[1] - qy
            if dx * dx + dy * dy < min_inset_sep_sq:
                return True
        return False

    for (ci, cj, v1, v2) in boundaries:
        port_specs = _validated_port_positions(
            v1, v2,
            partitions[ci].polygon,
            partitions[cj].polygon,
            robot_radius,
            checker,
        )
        for midpoint, inset_i, inset_j in port_specs:
            if _too_close_to_existing(ci, inset_i):
                continue
            if _too_close_to_existing(cj, inset_j):
                continue
            node = f"port_{port_id}"
            G.add_node(node, kind="port", cell_pair=(ci, cj))
            node_positions[node] = midpoint
            cell_incident[ci].append(node)
            cell_incident[cj].append(node)
            cell_boundary_ports[ci][port_id] = inset_i
            cell_boundary_ports[cj][port_id] = inset_j
            per_cell_insets[ci].append(inset_i)
            per_cell_insets[cj].append(inset_j)
            port_id += 1

    # ---- Robot start nodes ----
    start_cells: Dict[int, int] = {}
    for r, (sx, sy) in enumerate(robot_starts):
        node = f"start_{r}"
        ci = _find_containing_cell(partitions, sx, sy)
        if ci is None:
            continue
        G.add_node(node, kind="start", robot=r)
        node_positions[node] = (sx, sy)
        cell_incident[ci].append(node)
        start_cells[r] = ci

    # ---- Robot goal nodes ----
    goal_cells: Dict[int, int] = {}
    for r, (gx, gy) in enumerate(robot_goals):
        node = f"goal_{r}"
        ci = _find_containing_cell(partitions, gx, gy)
        if ci is None:
            continue
        G.add_node(node, kind="goal", robot=r)
        node_positions[node] = (gx, gy)
        cell_incident[ci].append(node)
        goal_cells[r] = ci

    # ---- Wire edges according to topology ----
    if topology == "pairwise":
        _add_pairwise_edges(G, partitions, cell_incident, node_positions)
    else:
        _add_star_edges(G, partitions, cell_incident, node_positions)

    return HighLevelGraph(
        graph=G,
        node_positions=node_positions,
        cell_boundary_ports=cell_boundary_ports,
        cell_incident_nodes=cell_incident,
        start_cells=start_cells,
        goal_cells=goal_cells,
        topology=topology,
    )


# ---------------------------------------------------------------------------
# Time-horizon estimation
# ---------------------------------------------------------------------------

def estimate_time_horizon(
        hlg: HighLevelGraph,
        congestion_factor: float = 2.0,
        min_horizon: int = 10,
) -> int:
    """Auto-compute time horizon from shortest hop-counts.

    For each robot, compute the unweighted shortest-path length (number of
    hops) from its start node to its goal node in the high-level graph.
    The time horizon is ``max(min_horizon, congestion_factor × max_hops)``.

    This mirrors ``multi_robot_flow_solver._estimate_time_horizon`` which
    uses ``max(10, manhattan_distance × 2)``.
    """
    max_hops = 0
    for r in hlg.start_cells:
        src = f"start_{r}"
        dst = f"goal_{r}"
        if src not in hlg.graph or dst not in hlg.graph:
            continue
        try:
            hops = nx.shortest_path_length(hlg.graph, src, dst)
            max_hops = max(max_hops, hops)
        except nx.NetworkXNoPath:
            continue
    return max(min_horizon, int(congestion_factor * max_hops))


