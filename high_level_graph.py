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
Each shared edge produces one **port** node at the edge's midpoint.  The
``cell_boundary_ports`` output maps each cell index to a
``{boundary_id: (x, y)}`` dict that can be passed directly to
``cell_joint_prm.build_cell_roadmap`` so the per-cell PRM anchors a joint
configuration at every port.

Time-horizon estimation
-----------------------
``estimate_time_horizon`` counts the maximum unweighted shortest-path
length (in hops) from any robot's start node to its goal node and
multiplies by a congestion factor, matching the heuristic in
``multi_robot_flow_solver.py:224-236``.

PRM-based pruning
-----------------
``prune_by_prm`` removes edges between port nodes whose corresponding PRM
port configurations are disconnected inside the cell roadmap, implementing
the "static cost + PRM feasibility" hybrid described in
``docs/design_decisions.md``.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2, Pol2

from partition import Partition

if TYPE_CHECKING:
    from cell_joint_prm import CellRoadmap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Round vertex coordinates to this many decimal places when comparing edges
# across cells.  Vertices come from _face_to_polygon's a0() projection so
# they should be bit-identical; rounding guards against float noise.
_COORD_PRECISION = 10


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
        For each cell index, the ``{boundary_id: (x, y)}`` mapping that
        :func:`cell_joint_prm.build_cell_roadmap` expects.
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
) -> List[Tuple[int, int, float, float]]:
    """Discover edges shared between partition polygons.

    Returns a list of ``(cell_i, cell_j, midpoint_x, midpoint_y)`` tuples,
    one per shared edge.  ``cell_i < cell_j`` always.
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

    boundaries: List[Tuple[int, int, float, float]] = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            shared = cell_edge_sets[i] & cell_edge_sets[j]
            for edge in shared:
                v1, v2 = edge
                mx = (v1[0] + v2[0]) / 2.0
                my = (v1[1] + v2[1]) / 2.0
                boundaries.append((i, j, mx, my))
    return boundaries


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
        Disc robot radius (used only for future extensions; the cell
        density already encodes the radius).
    topology :
        ``"pairwise"`` — O(m²) edges per cell, one hop per traversal.
        ``"star"`` — O(m) edges per cell, two hops per traversal.
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
    boundaries = _find_shared_edges(partitions)
    for port_id, (ci, cj, mx, my) in enumerate(boundaries):
        node = f"port_{port_id}"
        G.add_node(node, kind="port", cell_pair=(ci, cj))
        node_positions[node] = (mx, my)
        cell_incident[ci].append(node)
        cell_incident[cj].append(node)
        cell_boundary_ports[ci][port_id] = (mx, my)
        cell_boundary_ports[cj][port_id] = (mx, my)

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


# ---------------------------------------------------------------------------
# PRM-based edge pruning
# ---------------------------------------------------------------------------

def prune_by_prm(
        hlg: HighLevelGraph,
        cell_roadmaps: List["CellRoadmap"],
) -> int:
    """Remove port–port edges whose PRM port nodes are disconnected.

    Implements the hybrid cost strategy: edge *costs* stay Euclidean, but
    edges that cannot be realised inside the cell roadmap are deleted.

    Only prunes edges between two port nodes that share the same cell.
    Start/goal node reachability is verified during path extraction (step 7).

    Returns the number of edges removed.
    """
    to_remove = []
    for u, v, data in hlg.graph.edges(data=True):
        ci = data["cell_id"]
        if ci >= len(cell_roadmaps):
            continue
        rm = cell_roadmaps[ci]

        u_kind = hlg.graph.nodes[u].get("kind")
        v_kind = hlg.graph.nodes[v].get("kind")
        if u_kind != "port" or v_kind != "port":
            continue

        u_bid = _node_boundary_id(u)
        v_bid = _node_boundary_id(v)
        if u_bid is None or v_bid is None:
            continue

        u_cfg = rm.port_nodes.get(u_bid)
        v_cfg = rm.port_nodes.get(v_bid)
        if u_cfg is None or v_cfg is None:
            to_remove.append((u, v))
            continue
        if not nx.has_path(rm.graph, u_cfg, v_cfg):
            to_remove.append((u, v))

    for u, v in to_remove:
        hlg.graph.remove_edge(u, v)
    return len(to_remove)


def _node_boundary_id(node_name: str) -> Optional[int]:
    """Extract the integer boundary_id from ``'port_3'``."""
    if node_name.startswith("port_"):
        try:
            return int(node_name.split("_", 1)[1])
        except ValueError:
            return None
    return None
