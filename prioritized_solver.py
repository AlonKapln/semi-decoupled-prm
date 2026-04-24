"""Prioritized space-time A* on the high-level graph (Silver's
Cooperative A*, 2005). Robots are planned longest-first against a
shared reservation table that enforces vertex, swap, and per-cell
capacity conflicts. Returns None if any robot is unroutable.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from high_level_graph import HighLevelGraph

# Per-robot time-indexed node sequence.
RoutingSolution = Dict[int, List[str]]


def solve_prioritized(
        hlg: HighLevelGraph,
        num_robots: int,
        time_horizon: int,
        verbose: bool = False,
) -> Optional[RoutingSolution]:
    G = hlg.graph
    capacity_by_cell = _get_capacity_by_cell(hlg)
    node_to_cells = _build_node_to_cells(hlg)

    # Priority order: plan the longest start-to-goal path first.
    priorities: List[Tuple[int, int]] = []
    for r in range(num_robots):
        try:
            dist = nx.shortest_path_length(G, f"start_{r}", f"goal_{r}")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = time_horizon  # unreachable: plan first so we fail fast
        priorities.append((dist, r))
    priorities.sort(reverse=True)
    robot_order = [r for _, r in priorities]

    if verbose:
        print(f"Prioritized planner: order = {robot_order}")

    reservation: Dict[str, Dict[int, Set[int]]] = {}
    solution: RoutingSolution = {}

    for r in robot_order:
        start_node = f"start_{r}"
        goal_node = f"goal_{r}"
        if start_node not in G or goal_node not in G:
            if verbose:
                print(f"Robot {r}: start or goal not in graph")
            return None

        path = _astar_time_expanded(
            G, start_node, goal_node, time_horizon,
            reservation, node_to_cells, capacity_by_cell, hlg,
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
    """Invert cell_incident_nodes -> {node: [cell_indices]}."""
    mapping: Dict[str, List[int]] = {}
    for ci, nodes in hlg.cell_incident_nodes.items():
        for n in nodes:
            mapping.setdefault(n, []).append(ci)
    return mapping


def _get_capacity_by_cell(hlg: HighLevelGraph) -> Dict[int, int]:
    """Cell index -> density, read from edge attributes."""
    capacity_by_cell: Dict[int, int] = {}
    for _, _, data in hlg.graph.edges(data=True):
        capacity_by_cell[data["cell_id"]] = data["capacity"]
    return capacity_by_cell


def _capacity_ok(
        node: str,
        time: int,
        reservation: Dict[str, Dict[int, Set[int]]],
        node_to_cells: Dict[str, List[int]],
        densities: Dict[int, int],
        hlg: HighLevelGraph,
) -> bool:
    """True iff placing one more robot at (node, time) fits every incident
    cell's capacity."""
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
        hlg: HighLevelGraph,
) -> bool:
    """True iff from_node -> to_node at `time` collides with a robot going
    the other way. Uses node_equivalence so swaps between geometrically
    coincident names (e.g. start_i == goal_j) are caught."""
    equiv = hlg.node_equivalence
    robots_at_to = set()
    for n in equiv.get(to_node, [to_node]):
        robots_at_to |= reservation.get(n, {}).get(time, set())
    if not robots_at_to:
        return False
    robots_at_from_next = set()
    for n in equiv.get(from_node, [from_node]):
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
) -> Optional[List[str]]:
    """A* on (node, time) with vertex/swap/capacity conflicts.

    Edge cost is the Euclidean `cost` attribute; the heuristic is
    Dijkstra-to-goal on the untimed graph (admissible because realised
    segments are never shorter than the straight-line distance).
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
    # First-push came_from would record a suboptimal predecessor under
    # a consistent heuristic; overwrite on any strictly-improving push.
    g_score: Dict[Tuple[str, int], float] = {(start, 0): 0.0}

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
                    for en in hlg.node_equivalence.get(next_node, [next_node])
            ):
                continue

            if next_node != node and _swap_conflict(
                    node, next_node, time, reservation, hlg,
            ):
                continue

            if next_node == node:
                step_cost = 0.0  # holding in place is free
            else:
                step_cost = G.edges[node, next_node].get("cost", 1.0)
            new_g = g + step_cost
            next_state = (next_node, next_time)
            if new_g >= g_score.get(next_state, float("inf")):
                continue
            g_score[next_state] = new_g
            came_from[next_state] = (node, time)
            h = dist_to_goal.get(next_node, float("inf"))
            counter += 1
            heapq.heappush(
                pq, (new_g + h, new_g, counter, next_node, next_time),
            )

    return None
