"""Multi-commodity flow solver on the high-level cell graph.

Step 6 of ``plan.tex``. Given a :class:`HighLevelGraph` from step 5 and
a time horizon ``T``, route every robot from its start node to its goal
node subject to per-cell capacity constraints.

Strategy
--------
Sequential A* on a time-expanded graph with a reservation table. Robots
are planned one at a time in decreasing order of their start→goal graph
distance (longest path first) so scarce corridor capacity goes to the
robots that need it most. Each planned robot adds its time-indexed
occupancy to the reservation table; subsequent A* searches reject any
``(node, time)`` state that would violate vertex/swap conflicts or a
per-cell capacity bound.

Output
------
``MCFSolution = Dict[int, List[str]]`` mapping each robot index to its
time-indexed node sequence ``[node_at_t0, …, node_at_T]``. Returns
``None`` if any robot is unroutable.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from high_level_graph import HighLevelGraph

# Per-robot time-indexed node sequence.
MCFSolution = Dict[int, List[str]]


def solve_mcf(
        hlg: HighLevelGraph,
        num_robots: int,
        time_horizon: int,
        verbose: bool = False,
) -> Optional[MCFSolution]:
    """Route every robot on the high-level graph. See module docstring."""
    G = hlg.graph
    densities = _cell_densities(hlg)
    node_to_cells = _build_node_to_cells(hlg)
    node_equiv = _build_node_equivalence(hlg)

    # Priority order: plan longest start→goal path first.
    priorities: List[Tuple[int, int]] = []
    for r in range(num_robots):
        try:
            dist = nx.shortest_path_length(G, f"start_{r}", f"goal_{r}")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = time_horizon  # unreachable → plan first so we fail fast
        priorities.append((dist, r))
    priorities.sort(reverse=True)
    robot_order = [r for _, r in priorities]

    if verbose:
        print(f"MCF priority order: {robot_order}")

    reservation: Dict[str, Dict[int, Set[int]]] = {}
    solution: MCFSolution = {}

    for r in robot_order:
        start_node = f"start_{r}"
        goal_node = f"goal_{r}"
        if start_node not in G or goal_node not in G:
            if verbose:
                print(f"Robot {r}: start or goal not in graph")
            return None

        path = _astar_time_expanded(
            G, start_node, goal_node, time_horizon,
            reservation, node_to_cells, densities, hlg, node_equiv,
        )
        if path is None:
            if verbose:
                print(f"Robot {r}: no path found")
            return None

        if verbose:
            moves = sum(
                1 for i in range(len(path) - 1) if path[i] != path[i + 1]
            )
            arrive = next(
                (t for t, n in enumerate(path) if n == goal_node), len(path),
            )
            print(f"Robot {r}: {moves} moves, arrives at t={arrive}")

        solution[r] = path
        for t, n in enumerate(path):
            reservation.setdefault(n, {}).setdefault(t, set()).add(r)

    return solution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_node_to_cells(hlg: HighLevelGraph) -> Dict[str, List[int]]:
    """Invert cell_incident_nodes → {node: [cell_indices]}."""
    mapping: Dict[str, List[int]] = {}
    for ci, nodes in hlg.cell_incident_nodes.items():
        for n in nodes:
            mapping.setdefault(n, []).append(ci)
    return mapping


def _cell_densities(hlg: HighLevelGraph) -> Dict[int, int]:
    """Cell index → density. Read from graph edge attributes."""
    densities: Dict[int, int] = {}
    for _, _, data in hlg.graph.edges(data=True):
        densities[data["cell_id"]] = data["capacity"]
    return densities


def _build_node_equivalence(
        hlg: HighLevelGraph,
) -> Dict[str, List[str]]:
    """Group nodes by geometric position.

    Two distinct HLG nodes at the same coordinate (e.g. ``start_i`` and
    ``goal_j`` when two robots swap endpoints) are geometrically one
    location, so vertex/swap conflict checks must treat them as
    interchangeable.
    """
    pos_to_nodes: Dict[Tuple[float, float], List[str]] = {}
    for n, pos in hlg.node_positions.items():
        key = (round(pos[0], 6), round(pos[1], 6))
        pos_to_nodes.setdefault(key, []).append(n)
    return {
        n: pos_to_nodes[(round(pos[0], 6), round(pos[1], 6))]
        for n, pos in hlg.node_positions.items()
    }


def _capacity_ok(
        node: str,
        time: int,
        reservation: Dict[str, Dict[int, Set[int]]],
        node_to_cells: Dict[str, List[int]],
        densities: Dict[int, int],
        hlg: HighLevelGraph,
) -> bool:
    """True iff placing one more robot at ``node`` at ``time`` fits every
    incident cell's capacity."""
    for ci in node_to_cells.get(node, []):
        cap = densities.get(ci)
        if cap is None:
            continue
        count = sum(
            len(reservation.get(n, {}).get(time, set()))
            for n in hlg.cell_incident_nodes.get(ci, [])
        )
        if count >= cap:
            return False
    return True


def _swap_conflict(
        from_node: str,
        to_node: str,
        time: int,
        reservation: Dict[str, Dict[int, Set[int]]],
        node_equiv: Dict[str, List[str]],
) -> bool:
    """True iff moving ``from_node → to_node`` at ``time`` collides with a
    robot moving the opposite direction on the same edge.

    Uses position equivalence so geometric swaps are caught even when
    the two nodes are technically different (e.g. ``start_i`` == ``goal_j``).
    """
    robots_at_to = set()
    for n in node_equiv.get(to_node, [to_node]):
        robots_at_to |= reservation.get(n, {}).get(time, set())
    if not robots_at_to:
        return False
    robots_at_from_next = set()
    for n in node_equiv.get(from_node, [from_node]):
        robots_at_from_next |= reservation.get(n, {}).get(time + 1, set())
    return bool(robots_at_to & robots_at_from_next)


def _astar_time_expanded(
        G: nx.Graph,
        start: str,
        goal: str,
        T: int,
        reservation: Dict[str, Dict[int, Set[int]]],
        node_to_cells: Dict[str, List[int]],
        densities: Dict[int, int],
        hlg: HighLevelGraph,
        node_equiv: Dict[str, List[str]],
) -> Optional[List[str]]:
    """A* on ``(node, time)`` with vertex/swap conflicts and capacity.

    Uses kinematic edge cost (``cost`` attribute = Euclidean distance
    between HLG node positions) for both the g-score step and the h
    estimate (``single_source_dijkstra_path_length`` on the underlying
    HLG, which is admissible because each realised segment is at least
    as long as the straight-line distance).
    """
    try:
        dist_to_goal = nx.single_source_dijkstra_path_length(
            G, goal, weight="cost",
        )
    except nx.NetworkXError:
        return None

    counter = 0
    pq: List[Tuple] = [
        (dist_to_goal.get(start, float("inf")), 0.0, counter, start, 0),
    ]
    visited: Set[Tuple[str, int]] = set()
    came_from: Dict[Tuple[str, int], Tuple[str, int]] = {}

    while pq:
        _f, g, _, node, time = heapq.heappop(pq)

        if node == goal:
            path: List[str] = []
            cur = (node, time)
            while cur in came_from:
                path.append(cur[0])
                cur = came_from[cur]
            path.append(cur[0])
            path.reverse()
            while len(path) < T + 1:
                path.append(goal)
            return path

        state = (node, time)
        if state in visited:
            continue
        visited.add(state)

        if time >= T:
            continue

        next_time = time + 1
        for next_node in [node, *G.neighbors(node)]:
            if (next_node, next_time) in visited:
                continue

            if not _capacity_ok(
                    next_node, next_time, reservation,
                    node_to_cells, densities, hlg,
            ):
                continue

            # Vertex conflict (position-equivalent nodes count).
            if any(
                    reservation.get(en, {}).get(next_time, set())
                    for en in node_equiv.get(next_node, [next_node])
            ):
                continue

            if next_node != node and _swap_conflict(
                    node, next_node, time, reservation, node_equiv,
            ):
                continue

            if next_node == node:
                step_cost = 0.0  # hold — no geometric motion
            else:
                step_cost = G.edges[node, next_node].get("cost", 1.0)
            new_g = g + step_cost
            h = dist_to_goal.get(next_node, float("inf"))
            counter += 1
            heapq.heappush(
                pq, (new_g + h, new_g, counter, next_node, next_time),
            )
            if (next_node, next_time) not in came_from:
                came_from[(next_node, next_time)] = (node, time)

    return None
