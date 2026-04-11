"""High-level cell-adjacency graph for multi-commodity flow.

Step 5 of ``plan.tex``. Given the partitions from step 2 and robot
start/goal positions, this module builds a graph whose

- **Nodes** are validated boundary ports (on shared edges between cells)
  plus robot start and goal positions.
- **Edges** represent traversal through a cell, connecting every pair of
  nodes incident to that cell (pairwise topology). Edge attributes
  include the cell index, capacity, and Euclidean cost.

Adjacency discovery
-------------------
Two cells are adjacent when their ``Pol2.Polygon_2`` outlines share a
complete edge (a pair of consecutive vertices). Vertex coordinates are
rounded to ``_COORD_PRECISION`` decimal places to absorb any float noise
from the ``a0()`` rational projection.

Boundary-port placement
-----------------------
Each shared edge produces zero or more port nodes along subsegments
where an ``r``-inward inset is valid on both sides **and** survives a
``2r`` straight-line escape check (so ports never sit in dead-end
pockets). Long edges get multiple ports spaced by
``_MIN_PORT_SPACING_R * r`` so MCF sees parallel capacity channels
instead of bottlenecking the wall on a single transit slot. Ports are
also kept ``_PORT_EDGE_END_MARGIN_R * r`` away from edge endpoints, and
per-cell proximity tracking rejects any candidate whose inset in either
adjacent cell lands within that same minimum spacing of an
already-placed inset.

The ``cell_boundary_ports`` output maps each cell index to a ``{port_id:
inset_position_inside_that_cell}`` dict: the per-side inset is
precomputed at HLG build time so path realisation can look it up
directly without rerunning the geometry search.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2, Pol2

from partition import Partition

# Round vertex coordinates to this many decimal places when comparing
# edges across cells. Vertices come from _face_to_polygon's a0()
# projection so they should be bit-identical; rounding guards against
# float noise.
_COORD_PRECISION = 10

# Shift port insets along the shared-edge tangent (in opposite
# directions per side) so opposing insets are not directly perpendicular
# across from each other. Without this, both sides inset by exactly r
# land 2r apart and discopygal's swept-disc collision check reports
# tangent contact between adjacent-cell robots as a collision.
_PORT_TANGENT_OFFSET = 0.2

# Minimum along-edge spacing (in units of robot radius) between adjacent
# ports on the same shared edge. Same constant is reused for the
# cross-edge proximity check so two ports on different shared edges of
# the same cell are never closer in cell interior than this threshold.
# Below ~8r the ad-hoc joint PRM struggles to find intermediate configs
# for two transit robots near each other; above ~16r MCF
# under-utilises long walls. 10r is the sweet spot.
_MIN_PORT_SPACING_R = 10.0

# Margin (in units of robot radius) kept between any port and the
# endpoints of its shared edge. Different shared edges of the same cell
# meet at such endpoints, so trimming the valid region at each end
# pushes ports away from corners and complements the per-cell proximity
# check above.
_PORT_EDGE_END_MARGIN_R = 3.0


@dataclass
class HighLevelGraph:
    """Result of :func:`build_high_level_graph`.

    Attributes
    ----------
    graph : nx.Graph
        Nodes are strings (``"port_0"``, ``"start_1"``, ``"goal_2"``)
        with attribute ``kind``. Edges carry ``cell_id`` (int),
        ``capacity`` (int — per-cell density), and ``cost`` (float —
        Euclidean distance between endpoints).
    node_positions : dict[str, (float, float)]
        Geometric position for every node.
    cell_boundary_ports : dict[int, dict[int, (float, float)]]
        For each cell index, ``{port_id: (x, y)}`` where ``(x, y)`` is
        the precomputed inset of the port *inside that cell*. Used
        directly by path realisation as the joint-PRM entry/exit
        position for a robot transiting the cell via that port.
    cell_incident_nodes : dict[int, list[str]]
        For each cell index, the list of node IDs incident to it
        (ports, starts, goals).
    start_cells : dict[int, int]
        Robot index → cell index containing the robot's start.
    goal_cells : dict[int, int]
        Robot index → cell index containing the robot's goal.
    """

    graph: nx.Graph
    node_positions: Dict[str, Tuple[float, float]]
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]]
    cell_incident_nodes: Dict[int, List[str]]
    start_cells: Dict[int, int]
    goal_cells: Dict[int, int]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_in_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
    """Closed (boundary-inclusive) containment test."""
    return poly.bounded_side(
        Point_2(FT(x), FT(y)),
    ) != Bounded_side.ON_UNBOUNDED_SIDE


def _find_shared_edges(
        partitions: List[Partition],
) -> List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]:
    """Discover edges shared between partition polygons.

    Returns ``(cell_i, cell_j, v1, v2)`` tuples with ``cell_i < cell_j``
    and canonical (sorted) vertex ordering.
    """
    cell_edge_sets: List[set] = []
    for p in partitions:
        verts = [
            (
                round(v.x().to_double(), _COORD_PRECISION),
                round(v.y().to_double(), _COORD_PRECISION),
            )
            for v in p.polygon.vertices()
        ]
        n = len(verts)
        cell_edge_sets.append({
            tuple(sorted([verts[i], verts[(i + 1) % n]]))
            for i in range(n)
        })

    boundaries: List[
        Tuple[int, int, Tuple[float, float], Tuple[float, float]]
    ] = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            for v1, v2 in cell_edge_sets[i] & cell_edge_sets[j]:
                boundaries.append((i, j, v1, v2))
    return boundaries


def _edge_inward_normal(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        poly: Pol2.Polygon_2,
) -> Optional[Tuple[float, float]]:
    """Unit vector from the edge midpoint into the cell interior, or None."""
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

    Tries a decreasing sequence of fractions of ``r``. Allowing sub-``r``
    insets handles decomposition artifacts — thin slivers narrower than
    ``r`` that a robot physically transits without ever needing 2r of
    clearance on both sides. ``_has_2r_escape`` catches dead-end pockets
    separately.
    """
    for frac in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.025, 0.01):
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
    """True iff a 2r-length straight line from ``(x, y)`` stays valid.

    Tests ``n_dirs`` evenly spaced directions. Distinguishes dead-end
    pockets (no 2r line fits anywhere — e.g. a triangle wedged above an
    inflated-obstacle corner) from narrow decomposition slivers whose
    lengthwise direction easily admits a 2r straight line.
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
    """For each sample index along the edge, the tangent-shifted
    ``(inset_i, inset_j)`` pair if valid, else ``None``.

    Validity requires per-side ``_max_valid_inset`` success, the
    tangent-shifted variant staying inside both polygons and
    checker-valid, and both insets admitting a 2r straight-line escape.
    """
    ni = _edge_inward_normal(v1, v2, poly_i)
    nj = _edge_inward_normal(v1, v2, poly_j)
    if ni is None or nj is None:
        return [None] * samples

    tlen = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    tx = (v2[0] - v1[0]) / tlen if tlen > 1e-12 else 0.0
    ty = (v2[1] - v1[1]) / tlen if tlen > 1e-12 else 0.0
    off = robot_radius * _PORT_TANGENT_OFFSET

    def _ok(poly, pt):
        return _point_in_polygon(poly, pt[0], pt[1]) and (
            checker is None
            or checker.is_point_valid(Point_2(FT(pt[0]), FT(pt[1])))
        )

    entries: List[
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    ] = [None] * samples
    for k in range(samples):
        t = k / (samples - 1)
        px = v1[0] + t * (v2[0] - v1[0])
        py = v1[1] + t * (v2[1] - v1[1])
        pi = _max_valid_inset(px, py, ni[0], ni[1], poly_i, robot_radius, checker)
        if pi is None:
            continue
        pj = _max_valid_inset(px, py, nj[0], nj[1], poly_j, robot_radius, checker)
        if pj is None:
            continue

        # Tangent offset: shift pi and pj in opposite directions along
        # the shared edge so the two insets are NOT directly across
        # from each other. Without this, both sides inset by exactly r
        # along the perpendicular and sit exactly 2r apart, which
        # discopygal's verify_paths reports as a tangent-disc collision
        # when two robots cross the edge simultaneously.
        if off > 0.0:
            pi_shift = (pi[0] + tx * off, pi[1] + ty * off)
            pj_shift = (pj[0] - tx * off, pj[1] - ty * off)
            if _ok(poly_i, pi_shift) and _ok(poly_j, pj_shift):
                pi, pj = pi_shift, pj_shift

        if not _has_2r_escape(pi[0], pi[1], poly_i, robot_radius, checker):
            continue
        if not _has_2r_escape(pj[0], pj[1], poly_j, robot_radius, checker):
            continue
        entries[k] = (pi, pj)
    return entries


def _contiguous_runs(
        entries: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` (inclusive) index pairs for each maximal
    run of non-``None`` sample indices."""
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
    cells share a wall many times the robot radius — benefit from
    multiple parallel ports: MCF sees each port as an independent
    transit slot, so per-timestep flow across that wall is no longer
    bottlenecked by a single entry point.

    Steps: densely sample the edge, group into maximal contiguous runs
    of valid samples, trim each run by the corner margin, and
    distribute ports within each run at the minimum along-edge spacing.

    Returns ``(midpoint, inset_i, inset_j)`` tuples — possibly empty.
    """
    edge_len = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    if edge_len < 1e-12:
        return []

    samples = max(21, int(edge_len / (0.5 * robot_radius)) + 1)
    entries = _sample_valid_insets(
        v1, v2, poly_i, poly_j, robot_radius, checker, samples,
    )

    # Trim by end margin so ports never sit on an edge endpoint. Short
    # edges still admit a centre port: cap the margin at 40% of length.
    margin = min(_PORT_EDGE_END_MARGIN_R * robot_radius, 0.4 * edge_len)
    t_margin = margin / edge_len
    k_margin_start = int(math.ceil(t_margin * (samples - 1)))
    k_margin_end = int((1.0 - t_margin) * (samples - 1))

    min_spacing = _MIN_PORT_SPACING_R * robot_radius
    out: List[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = []

    for (s, e) in _contiguous_runs(entries):
        s = max(s, k_margin_start)
        e = min(e, k_margin_end)
        if s > e:
            continue
        t_s = s / (samples - 1)
        t_e = e / (samples - 1)
        run_len = edge_len * (t_e - t_s)
        # Ports fit: n points with n-1 gaps of >= min_spacing.
        n_ports = max(1, 1 + int(run_len // min_spacing))
        for p in range(n_ports):
            t_target = (
                (t_s + t_e) / 2.0 if n_ports == 1
                else t_s + (t_e - t_s) * (p / (n_ports - 1))
            )
            k = max(s, min(e, round(t_target * (samples - 1))))
            pair = entries[k]
            if pair is None:
                continue
            inset_i, inset_j = pair
            t_mid = k / (samples - 1)
            midpoint = (
                v1[0] + t_mid * (v2[0] - v1[0]),
                v1[1] + t_mid * (v2[1] - v1[1]),
            )
            out.append((midpoint, inset_i, inset_j))
    return out


def _find_containing_cell(
        partitions: List[Partition], x: float, y: float,
) -> Optional[int]:
    """Index of the first partition containing ``(x, y)``."""
    for idx, p in enumerate(partitions):
        if _point_in_polygon(p.polygon, x, y):
            return idx
    return None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_high_level_graph(
        partitions: List[Partition],
        robot_starts: List[Tuple[float, float]],
        robot_goals: List[Tuple[float, float]],
        robot_radius: float,
        checker=None,
) -> HighLevelGraph:
    """Build the high-level cell graph for MCF routing.

    Args
    ----
    partitions : free-space cells from step 2.
    robot_starts, robot_goals : ``(x, y)`` per robot.
    robot_radius : disc robot radius.
    checker : optional ``ObjectCollisionDetection`` used to validate port
        insets against the true (arc-exact) inflated obstacles.
    """
    G = nx.Graph()
    node_positions: Dict[str, Tuple[float, float]] = {}
    cell_incident: Dict[int, List[str]] = {
        i: [] for i in range(len(partitions))
    }
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]] = {
        i: {} for i in range(len(partitions))
    }

    # ---- Boundary port nodes ----
    # Tracks already-placed insets per cell so two ports on different
    # shared edges of the same cell can't land within the minimum
    # spacing of each other — that would break the joint PRM's 2r
    # pairwise separation at cell corners.
    per_cell_insets: Dict[int, List[Tuple[float, float]]] = {
        i: [] for i in range(len(partitions))
    }
    min_inset_sep_sq = (_MIN_PORT_SPACING_R * robot_radius) ** 2

    def _too_close(cell: int, point: Tuple[float, float]) -> bool:
        for qx, qy in per_cell_insets[cell]:
            dx = point[0] - qx
            dy = point[1] - qy
            if dx * dx + dy * dy < min_inset_sep_sq:
                return True
        return False

    port_id = 0
    for (ci, cj, v1, v2) in _find_shared_edges(partitions):
        for midpoint, inset_i, inset_j in _validated_port_positions(
            v1, v2,
            partitions[ci].polygon, partitions[cj].polygon,
            robot_radius, checker,
        ):
            if _too_close(ci, inset_i) or _too_close(cj, inset_j):
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

    # ---- Robot start / goal nodes ----
    start_cells: Dict[int, int] = {}
    for r, (sx, sy) in enumerate(robot_starts):
        ci = _find_containing_cell(partitions, sx, sy)
        if ci is None:
            continue
        node = f"start_{r}"
        G.add_node(node, kind="start", robot=r)
        node_positions[node] = (sx, sy)
        cell_incident[ci].append(node)
        start_cells[r] = ci

    goal_cells: Dict[int, int] = {}
    for r, (gx, gy) in enumerate(robot_goals):
        ci = _find_containing_cell(partitions, gx, gy)
        if ci is None:
            continue
        node = f"goal_{r}"
        G.add_node(node, kind="goal", robot=r)
        node_positions[node] = (gx, gy)
        cell_incident[ci].append(node)
        goal_cells[r] = ci

    # ---- Pairwise edges: every pair of nodes incident to the same cell ----
    for ci, nodes in cell_incident.items():
        cap = partitions[ci].density
        for a in range(len(nodes)):
            ax, ay = node_positions[nodes[a]]
            for b in range(a + 1, len(nodes)):
                bx, by = node_positions[nodes[b]]
                G.add_edge(
                    nodes[a], nodes[b],
                    cell_id=ci, capacity=cap,
                    cost=math.hypot(ax - bx, ay - by),
                )

    return HighLevelGraph(
        graph=G,
        node_positions=node_positions,
        cell_boundary_ports=cell_boundary_ports,
        cell_incident_nodes=cell_incident,
        start_cells=start_cells,
        goal_cells=goal_cells,
    )


def estimate_time_horizon(
        hlg: HighLevelGraph,
        congestion_factor: float = 2.0,
        min_horizon: int = 10,
) -> int:
    """Auto-compute time horizon from shortest hop-counts.

    For each robot, the unweighted shortest-path length (hops) from its
    start to its goal in the high-level graph. The returned horizon is
    ``max(min_horizon, congestion_factor * max_hops)``.
    """
    max_hops = 0
    for r in hlg.start_cells:
        src = f"start_{r}"
        dst = f"goal_{r}"
        if src not in hlg.graph or dst not in hlg.graph:
            continue
        try:
            max_hops = max(max_hops, nx.shortest_path_length(hlg.graph, src, dst))
        except nx.NetworkXNoPath:
            continue
    return max(min_horizon, int(congestion_factor * max_hops))
