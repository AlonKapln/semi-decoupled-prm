import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2, Pol2

from partition import Partition

_COORD_PRECISION = 10

# Min along-edge spacing between parallel ports (units of r).
_MIN_PORT_SPACING_R = 4.0

# Min separation between two port points on different shared edges of
# the same cell (units of r). 2r is the joint-PRM floor; 2.5r gives
# steering room.
_MIN_CROSS_EDGE_PORT_POINT_SEP_R = 2.5

# Port-point inset depth (units of r). Must be > 1; 2.1r puts opposing
# points 4.2r apart and keeps each 1.1r clear of the shared edge.
_PORT_POINT_DEPTH_R = 2.1

# Margin (in r) between a port and its shared-edge endpoints, capped at
# 40% of edge length so short edges still admit a centre port.
_PORT_EDGE_END_MARGIN_R = 5.0


@dataclass
class HighLevelGraph:
    graph: nx.Graph
    node_positions: Dict[str, Tuple[float, float]]
    # cell_id -> {port_id: (x, y)} port point on that cell's side.
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]]
    cell_incident_nodes: Dict[int, List[str]]
    start_cells: Dict[int, int]
    goal_cells: Dict[int, int]
    # node -> every node at the same rounded position (includes self).
    # Two distinct nodes can share a coordinate (e.g. start_i == goal_j
    # on a swap); router conflict checks treat them as one point.
    node_equivalence: Dict[str, List[str]]


def _point_in_polygon(poly: Pol2.Polygon_2, x: float, y: float) -> bool:
    """True iff (x, y) is inside or on the polygon boundary."""
    return poly.bounded_side(
        Point_2(FT(x), FT(y)),
    ) != Bounded_side.ON_UNBOUNDED_SIDE


def _find_shared_edges(
        partitions: List[Partition],
) -> List[Tuple[int, int, Tuple[float, float], Tuple[float, float]]]:
    """Return (cell_i, cell_j, v1, v2) tuples with cell_i < cell_j for
    every pair of partitions that shares a vertex-pair edge."""
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
    """Unit vector from edge midpoint into the cell interior, or None
    if the edge is degenerate or the polygon orientation can't be
    determined locally."""
    dx, dy = v2[0] - v1[0], v2[1] - v1[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    normal_x, normal_y = -dy / length, dx / length
    mx = (v1[0] + v2[0]) / 2.0
    my = (v1[1] + v2[1]) / 2.0
    eps = 1e-6
    if _point_in_polygon(poly, mx + eps * normal_x, my + eps * normal_y):
        return (normal_x, normal_y)
    if _point_in_polygon(poly, mx - eps * normal_x, my - eps * normal_y):
        return (-normal_x, -normal_y)
    return None


def _fixed_port_point(
        px: float,
        py: float,
        normal_x: float,
        normal_y: float,
        poly: Pol2.Polygon_2,
        robot_radius: float,
        checker,
) -> Optional[Tuple[float, float]]:
    """Port point at depth _PORT_POINT_DEPTH_R * r, or None if it lands
    outside the polygon or fails the obstacle checker."""
    cx = px + _PORT_POINT_DEPTH_R * robot_radius * normal_x
    cy = py + _PORT_POINT_DEPTH_R * robot_radius * normal_y
    if not _point_in_polygon(poly, cx, cy):
        return None
    if checker is not None and not checker.is_point_valid(
            Point_2(FT(cx), FT(cy)),
    ):
        return None
    return (cx, cy)


def _has_2r_escape(
        x: float,
        y: float,
        poly: Pol2.Polygon_2,
        robot_radius: float,
        checker,
        n_dirs: int = 8,
        n_steps: int = 5,
) -> bool:
    """True iff some direction admits a 2r straight line from (x, y)
    that stays inside the polygon and clear of obstacles. Distinguishes
    dead-end pockets from narrow but traversable slivers.

    :param x: query x.
    :param y: query y.
    :param poly: cell polygon.
    :param robot_radius: disc radius r.
    :param checker: obstacle checker, or None.
    :param n_dirs: directions tested (uniform on [0, 2pi)).
    :param n_steps: discretisation per ray.
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


def _sample_valid_port_points(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        poly_i: Pol2.Polygon_2,
        poly_j: Pol2.Polygon_2,
        robot_radius: float,
        checker,
        samples: int,
        forbid_near_i: Optional[List[Tuple[float, float]]] = None,
        forbid_near_j: Optional[List[Tuple[float, float]]] = None,
) -> List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """For each of `samples` indices along the edge, the (point_i,
    point_j) pair if valid on both sides, else None.

    A pair is valid if both points are inside their cell, both pass the
    obstacle checker, both admit a 2r escape, and the line between them
    stays at least 2r from every anchor in either incident cell.

    :param v1: edge endpoint 1.
    :param v2: edge endpoint 2.
    :param poly_i: cell-i polygon.
    :param poly_j: cell-j polygon.
    :param robot_radius: disc radius r.
    :param checker: obstacle checker.
    :param samples: number of sample positions along the edge.
    :param forbid_near_i: anchors in cell i to keep the sweep clear of.
    :param forbid_near_j: anchors in cell j to keep the sweep clear of.
    :return: per-sample-index optional (point_i, point_j) pair.
    """
    ni = _edge_inward_normal(v1, v2, poly_i)
    nj = _edge_inward_normal(v1, v2, poly_j)
    if ni is None or nj is None:
        return [None] * samples

    anchor_min_dist_sq = (2.0 * robot_radius) ** 2
    anchors = list(forbid_near_i or []) + list(forbid_near_j or [])

    def _seg_min_dist_sq(
            px: float, py: float,
            ax: float, ay: float,
            bx: float, by: float,
    ) -> float:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-24:
            return (px - ax) ** 2 + (py - ay) ** 2
        t_ = ((px - ax) * dx + (py - ay) * dy) / L2
        t_ = max(0.0, min(1.0, t_))
        qx, qy = ax + t_ * dx, ay + t_ * dy
        return (px - qx) ** 2 + (py - qy) ** 2

    def _sweep_too_close(
            pi: Tuple[float, float], pj: Tuple[float, float],
    ) -> bool:
        for ax, ay in anchors:
            if _seg_min_dist_sq(
                    ax, ay, pi[0], pi[1], pj[0], pj[1],
            ) < anchor_min_dist_sq:
                return True
        return False

    entries: List[
        Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    ] = [None] * samples
    for k in range(samples):
        t = k / (samples - 1)
        px = v1[0] + t * (v2[0] - v1[0])
        py = v1[1] + t * (v2[1] - v1[1])
        pi = _fixed_port_point(px, py, ni[0], ni[1], poly_i, robot_radius, checker)
        if pi is None:
            continue
        pj = _fixed_port_point(px, py, nj[0], nj[1], poly_j, robot_radius, checker)
        if pj is None:
            continue
        if not _has_2r_escape(pi[0], pi[1], poly_i, robot_radius, checker):
            continue
        if not _has_2r_escape(pj[0], pj[1], poly_j, robot_radius, checker):
            continue
        if anchors and _sweep_too_close(pi, pj):
            continue
        entries[k] = (pi, pj)
    return entries


def _contiguous_runs(
        entries: List[Optional[Tuple[Tuple[float, float], Tuple[float, float]]]],
) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs for each maximal non-None run."""
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
        forbid_near_i: Optional[List[Tuple[float, float]]] = None,
        forbid_near_j: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Place one or more ports along a shared edge.

    Long edges get multiple parallel ports so the router sees independent
    transit slots instead of bottlenecking on one entry.

    :param v1: edge endpoint 1.
    :param v2: edge endpoint 2.
    :param poly_i: cell-i polygon.
    :param poly_j: cell-j polygon.
    :param robot_radius: disc radius r.
    :param checker: obstacle checker.
    :param forbid_near_i: anchors in cell i.
    :param forbid_near_j: anchors in cell j.
    :return: list of (midpoint, point_i, point_j) for each placed port.
    """
    edge_len = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    if edge_len < 1e-12:
        return []

    samples = max(21, int(edge_len / (0.5 * robot_radius)) + 1)
    entries = _sample_valid_port_points(
        v1, v2, poly_i, poly_j, robot_radius, checker, samples,
        forbid_near_i=forbid_near_i, forbid_near_j=forbid_near_j,
    )

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
            point_i, point_j = pair
            t_mid = k / (samples - 1)
            midpoint = (
                v1[0] + t_mid * (v2[0] - v1[0]),
                v1[1] + t_mid * (v2[1] - v1[1]),
            )
            out.append((midpoint, point_i, point_j))
    return out


def _find_containing_cell(
        partitions: List[Partition], x: float, y: float,
) -> Optional[int]:
    """Index of the first partition that contains (x, y), or None."""
    for idx, p in enumerate(partitions):
        if _point_in_polygon(p.polygon, x, y):
            return idx
    return None


def build_high_level_graph(
        partitions: List[Partition],
        robot_starts: List[Tuple[float, float]],
        robot_goals: List[Tuple[float, float]],
        robot_radius: float,
        checker=None,
) -> HighLevelGraph:
    """Build the cell-adjacency graph used by the router.

    Nodes are validated boundary ports plus per-robot starts and goals.
    Edges connect every pair of nodes incident to the same cell, carrying
    cell_id / capacity / cost.

    :param partitions: cells from the grid decomposition.
    :param robot_starts: per-robot start positions.
    :param robot_goals: per-robot goal positions.
    :param robot_radius: disc radius r.
    :param checker: optional ObjectCollisionDetection for arc-exact
        validation against inflated obstacles.
    :return: populated HighLevelGraph.
    """
    G = nx.Graph()
    node_positions: Dict[str, Tuple[float, float]] = {}
    cell_incident: Dict[int, List[str]] = {
        i: [] for i in range(len(partitions))
    }
    cell_boundary_ports: Dict[int, Dict[int, Tuple[float, float]]] = {
        i: {} for i in range(len(partitions))
    }

    # Per-cell list of port points already placed. Used to reject a new
    # port whose inset on this side lands within
    # _MIN_CROSS_EDGE_PORT_POINT_SEP_R of an existing one (would break
    # joint-PRM 2r separation at cell corners).
    per_cell_port_points: Dict[int, List[Tuple[float, float]]] = {
        i: [] for i in range(len(partitions))
    }
    min_port_point_sep_sq = (_MIN_CROSS_EDGE_PORT_POINT_SEP_R * robot_radius) ** 2

    def _too_close(cell: int, point: Tuple[float, float]) -> bool:
        for qx, qy in per_cell_port_points[cell]:
            dx = point[0] - qx
            dy = point[1] - qy
            if dx * dx + dy * dy < min_port_point_sep_sq:
                return True
        return False

    # Anchors keep ports clear of robot starts/goals. Boundary-inclusive
    # containment means a start exactly on a shared edge gets registered
    # in both cells.
    cell_anchors: Dict[int, List[Tuple[float, float]]] = {
        i: [] for i in range(len(partitions))
    }
    for sx, sy in robot_starts:
        for ci_anchor, p in enumerate(partitions):
            if _point_in_polygon(p.polygon, sx, sy):
                cell_anchors[ci_anchor].append((sx, sy))
    for gx, gy in robot_goals:
        for ci_anchor, p in enumerate(partitions):
            if _point_in_polygon(p.polygon, gx, gy):
                cell_anchors[ci_anchor].append((gx, gy))

    port_id = 0
    for (ci, cj, v1, v2) in _find_shared_edges(partitions):
        for midpoint, point_i, point_j in _validated_port_positions(
                v1, v2,
                partitions[ci].polygon, partitions[cj].polygon,
                robot_radius, checker,
                forbid_near_i=cell_anchors.get(ci),
                forbid_near_j=cell_anchors.get(cj),
        ):
            if _too_close(ci, point_i) or _too_close(cj, point_j):
                continue
            node = f"port_{port_id}"
            G.add_node(node, kind="port", cell_pair=(ci, cj))
            node_positions[node] = midpoint
            cell_incident[ci].append(node)
            cell_incident[cj].append(node)
            cell_boundary_ports[ci][port_id] = point_i
            cell_boundary_ports[cj][port_id] = point_j
            per_cell_port_points[ci].append(point_i)
            per_cell_port_points[cj].append(point_j)
            port_id += 1

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

    # Pairwise edges: every pair of nodes incident to the same cell.
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

    pos_to_nodes: Dict[Tuple[float, float], List[str]] = {}
    for n, pos in node_positions.items():
        key = (round(pos[0], 6), round(pos[1], 6))
        pos_to_nodes.setdefault(key, []).append(n)
    node_equivalence: Dict[str, List[str]] = {
        n: pos_to_nodes[(round(pos[0], 6), round(pos[1], 6))]
        for n, pos in node_positions.items()
    }

    return HighLevelGraph(
        graph=G,
        node_positions=node_positions,
        cell_boundary_ports=cell_boundary_ports,
        cell_incident_nodes=cell_incident,
        start_cells=start_cells,
        goal_cells=goal_cells,
        node_equivalence=node_equivalence,
    )


def estimate_time_horizon(
        hlg: HighLevelGraph,
        congestion_factor: float = 2.0,
        min_horizon: int = 10,
) -> int:
    """congestion_factor * (max start->goal hop count), floored at min_horizon."""
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
