"""Multi-commodity flow solver on the high-level cell graph.

Step 6 of ``plan.tex``.  Given a :class:`HighLevelGraph` from step 5 and a
time horizon ``T``, route every robot from its start node to its goal node
subject to per-cell capacity constraints.

Formulation
-----------
The solver works on a **time-expanded** copy of the high-level graph.  For
every node ``n`` and timestep ``t ∈ [0, T]`` there is a binary variable

    at[r][n][t] ∈ {0, 1}

indicating whether robot ``r`` occupies node ``n`` at time ``t``.

Constraints:

1. **Exactly-one** — each robot is at exactly one node per timestep::

       ∀ r, t :  Σ_n  at[r][n][t]  =  1

2. **Start** — robot ``r`` begins at its start node::

       at[r][start_r][0]  =  1

3. **Goal** — robot ``r`` ends at its goal node::

       at[r][goal_r][T]  =  1

4. **Conservation** — a robot can stay or move to a graph neighbour::

       ∀ r, n, t < T :
           at[r][n][t]  ≤  Σ_{m ∈ {n} ∪ Γ(n)}  at[r][m][t+1]

5. **Cell capacity** — at most ``density(c)`` robots may simultaneously
   occupy nodes incident to cell ``c``::

       ∀ c, t :  Σ_r  Σ_{n ∈ incident(c)}  at[r][n][t]  ≤  density(c)

   A port shared by two cells contributes to both — this is conservative
   (may reject feasible solutions) but safe (never over-packs a cell).

Objective
---------
Maximise the total time robots spend at their goals (equivalently: reach
goals as early as possible)::

    max  Σ_{r, t}  at[r][goal_r][t]

Three strategies
----------------
``"ilp"``
    Globally optimal via OR-Tools SCIP.  Practical for ≲10 robots.

``"sequential"``
    A* on the time-expanded graph with a reservation table.  Fast, scales
    well, but solution quality depends on planning order.

``"priority"``
    Like sequential but plans longest-path robots first, giving them
    priority for scarce corridor capacity.

Output
------
``MCFSolution = Dict[int, List[str]]`` mapping each robot index to its
time-indexed node sequence ``[node_at_t0, node_at_t1, …, node_at_T]``.
Returns ``None`` when no feasible routing exists.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from high_level_graph import HighLevelGraph

try:
    from ortools.linear_solver import pywraplp

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

# Per-robot time-indexed node sequence.
MCFSolution = Dict[int, List[str]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve_mcf(
    hlg: HighLevelGraph,
    num_robots: int,
    time_horizon: int,
    strategy: str = "ilp",
    verbose: bool = False,
) -> Optional[MCFSolution]:
    """Route robots on the high-level graph.

    Args
    ----
    hlg : HighLevelGraph from step 5.
    num_robots : number of robots to route.
    time_horizon : maximum number of timesteps ``T``.
    strategy : ``"ilp"``, ``"sequential"``, or ``"priority"``.
    verbose : print progress to stdout.

    Returns
    -------
    ``{robot_idx: [node_at_t0, …, node_at_T]}`` or ``None``.
    """
    if strategy == "ilp":
        return _solve_ilp(hlg, num_robots, time_horizon, verbose)
    if strategy == "sequential":
        return _solve_sequential(hlg, num_robots, time_horizon, verbose)
    if strategy == "priority":
        return _solve_priority(hlg, num_robots, time_horizon, verbose)
    raise ValueError(f"strategy must be 'ilp', 'sequential', or 'priority', got {strategy!r}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_node_to_cells(hlg: HighLevelGraph) -> Dict[str, List[int]]:
    """Invert cell_incident_nodes → {node: [cell_indices]}."""
    mapping: Dict[str, List[int]] = {}
    for ci, nodes in hlg.cell_incident_nodes.items():
        for n in nodes:
            mapping.setdefault(n, []).append(ci)
    return mapping


def _cell_densities(hlg: HighLevelGraph) -> Dict[int, int]:
    """Cell index → density.  Read from graph edge attributes."""
    densities: Dict[int, int] = {}
    for _, _, data in hlg.graph.edges(data=True):
        ci = data["cell_id"]
        cap = data["capacity"]
        densities[ci] = cap
    return densities


# ---------------------------------------------------------------------------
# ILP strategy
# ---------------------------------------------------------------------------

def _solve_ilp(
    hlg: HighLevelGraph,
    num_robots: int,
    T: int,
    verbose: bool,
) -> Optional[MCFSolution]:
    if not HAS_ORTOOLS:
        if verbose:
            print("OR-Tools not available, falling back to sequential solver")
        return _solve_sequential(hlg, num_robots, T, verbose)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        if verbose:
            print("SCIP solver not available")
        return None

    G = hlg.graph
    nodes = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    densities = _cell_densities(hlg)

    if verbose:
        print(f"ILP: {num_robots} robots, {N} nodes, T={T}")

    # --- Decision variables: at[r][n][t] ---
    at = {}
    for r in range(num_robots):
        at[r] = {}
        for ni in range(N):
            at[r][ni] = {}
            for t in range(T + 1):
                at[r][ni][t] = solver.BoolVar(f"at_r{r}_n{ni}_t{t}")

    # --- Objective: maximise time spent at goal ---
    objective = solver.Objective()
    for r in range(num_robots):
        goal_node = f"goal_{r}"
        if goal_node not in node_idx:
            continue
        gi = node_idx[goal_node]
        for t in range(T + 1):
            objective.SetCoefficient(at[r][gi][t], 1)
    objective.SetMaximization()

    # --- Constraint 1: exactly one node per robot per timestep ---
    for r in range(num_robots):
        for t in range(T + 1):
            ct = solver.Constraint(1, 1)
            for ni in range(N):
                ct.SetCoefficient(at[r][ni][t], 1)

    # --- Constraint 2: start conditions ---
    for r in range(num_robots):
        start_node = f"start_{r}"
        if start_node not in node_idx:
            return None  # robot has no start node in graph
        si = node_idx[start_node]
        solver.Add(at[r][si][0] == 1)

    # --- Constraint 3: goal conditions ---
    for r in range(num_robots):
        goal_node = f"goal_{r}"
        if goal_node not in node_idx:
            return None
        gi = node_idx[goal_node]
        solver.Add(at[r][gi][T] == 1)

    # --- Constraint 4: flow conservation ---
    for r in range(num_robots):
        for t in range(T):
            for ni in range(N):
                n = nodes[ni]
                reachable = [ni]  # can stay
                for nb in G.neighbors(n):
                    reachable.append(node_idx[nb])
                ct = solver.Constraint(-solver.infinity(), 0)
                ct.SetCoefficient(at[r][ni][t], -1)
                for mi in reachable:
                    ct.SetCoefficient(at[r][mi][t + 1], 1)

    # --- Constraint 5a: vertex conflicts ---
    # No two robots at the same node at the same timestep.
    for ni in range(N):
        for t in range(T + 1):
            ct = solver.Constraint(0, 1)
            for r in range(num_robots):
                ct.SetCoefficient(at[r][ni][t], 1)

    # --- Constraint 5b: no edge-swap conflicts ---
    # Two robots cannot swap positions on the same edge at the same timestep.
    # at[r1][u][t] + at[r1][v][t+1] + at[r2][v][t] + at[r2][u][t+1] <= 3
    for t in range(T):
        for u_name, v_name in G.edges():
            ui, vi = node_idx[u_name], node_idx[v_name]
            for r1 in range(num_robots):
                for r2 in range(r1 + 1, num_robots):
                    ct = solver.Constraint(-solver.infinity(), 3)
                    ct.SetCoefficient(at[r1][ui][t], 1)
                    ct.SetCoefficient(at[r1][vi][t + 1], 1)
                    ct.SetCoefficient(at[r2][vi][t], 1)
                    ct.SetCoefficient(at[r2][ui][t + 1], 1)

    # --- Constraint 6: per-cell capacity ---
    for ci, incident_nodes in hlg.cell_incident_nodes.items():
        cap = densities.get(ci)
        if cap is None:
            continue
        incident_indices = [
            node_idx[n] for n in incident_nodes if n in node_idx
        ]
        if not incident_indices:
            continue
        for t in range(T + 1):
            ct = solver.Constraint(0, cap)
            for r in range(num_robots):
                for ni in incident_indices:
                    ct.SetCoefficient(at[r][ni][t], 1)

    if verbose:
        print(
            f"ILP: {solver.NumVariables()} variables, "
            f"{solver.NumConstraints()} constraints"
        )
        print("Solving ILP...")

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        if verbose:
            print(f"No solution found (status={status})")
        return None

    if verbose:
        label = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        print(f"Solution found ({label})")

    # --- Extract solution ---
    solution: MCFSolution = {}
    for r in range(num_robots):
        path: List[str] = []
        for t in range(T + 1):
            for ni in range(N):
                if at[r][ni][t].solution_value() > 0.5:
                    path.append(nodes[ni])
                    break
            else:
                path.append(path[-1] if path else nodes[0])
        solution[r] = path

    return solution


# ---------------------------------------------------------------------------
# Sequential A* strategy
# ---------------------------------------------------------------------------

def _solve_sequential(
    hlg: HighLevelGraph,
    num_robots: int,
    T: int,
    verbose: bool,
    robot_order: Optional[List[int]] = None,
) -> Optional[MCFSolution]:
    """Plan robots one at a time with a reservation table."""
    G = hlg.graph
    densities = _cell_densities(hlg)
    node_to_cells = _build_node_to_cells(hlg)

    # Position-equivalence map: nodes at the same geometric position
    # (e.g. start_i and goal_j when robots swap endpoints) must be
    # treated as identical for vertex/swap conflict checks.
    pos_to_nodes: Dict[Tuple[float, float], List[str]] = {}
    for n, pos in hlg.node_positions.items():
        key = (round(pos[0], 6), round(pos[1], 6))
        pos_to_nodes.setdefault(key, []).append(n)
    node_equiv: Dict[str, List[str]] = {}
    for n, pos in hlg.node_positions.items():
        key = (round(pos[0], 6), round(pos[1], 6))
        node_equiv[n] = pos_to_nodes[key]

    if robot_order is None:
        robot_order = list(range(num_robots))

    # reservation[node][t] = set of robot indices occupying that node at t
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
            G, start_node, goal_node, T,
            reservation, node_to_cells, densities, hlg,
            node_equiv,
        )
        if path is None:
            if verbose:
                print(f"Robot {r}: no path found")
            return None

        if verbose:
            moves = sum(1 for i in range(len(path) - 1) if path[i] != path[i + 1])
            print(f"Robot {r}: {moves} moves, arrives at t={_first_goal_time(path, goal_node)}")

        solution[r] = path

        # Reserve this robot's path
        for t, n in enumerate(path):
            reservation.setdefault(n, {}).setdefault(t, set()).add(r)

    return solution


def _first_goal_time(path: List[str], goal: str) -> int:
    for t, n in enumerate(path):
        if n == goal:
            return t
    return len(path)


def _astar_time_expanded(
    G: nx.Graph,
    start: str,
    goal: str,
    T: int,
    reservation: Dict[str, Dict[int, Set[int]]],
    node_to_cells: Dict[str, List[int]],
    densities: Dict[int, int],
    hlg: HighLevelGraph,
    node_equiv: Optional[Dict[str, List[str]]] = None,
) -> Optional[List[str]]:
    """A* on (node, time) state space with capacity constraints."""

    # Precompute heuristic: shortest unweighted distance to goal
    try:
        dist_to_goal = nx.single_source_shortest_path_length(G, goal)
    except nx.NetworkXError:
        return None

    # Priority queue: (f_score, g_score, counter, node, time)
    counter = 0
    start_h = dist_to_goal.get(start, T)
    pq: List[Tuple] = [(start_h, 0, counter, start, 0)]
    visited: Set[Tuple[str, int]] = set()
    # came_from[(node, time)] = (prev_node, prev_time)
    came_from: Dict[Tuple[str, int], Tuple[str, int]] = {}

    while pq:
        f, g, _, node, time = heapq.heappop(pq)

        if node == goal:
            # Reconstruct path from came_from chain
            path: List[str] = []
            cur = (node, time)
            while cur in came_from:
                path.append(cur[0])
                cur = came_from[cur]
            path.append(cur[0])  # start node
            path.reverse()
            # Pad to T+1 (robot stays at goal)
            while len(path) < T + 1:
                path.append(goal)
            return path

        state = (node, time)
        if state in visited:
            continue
        visited.add(state)

        if time >= T:
            continue

        # Expand: stay or move to neighbours
        next_time = time + 1
        candidates = [node] + list(G.neighbors(node))

        for next_node in candidates:
            if (next_node, next_time) in visited:
                continue

            # Check capacity at next_node at next_time
            if not _capacity_ok(
                next_node, next_time, reservation,
                node_to_cells, densities, hlg,
            ):
                continue

            # Vertex conflict: no two robots at any geometrically-
            # equivalent node at the same time — geometrically they'd
            # overlap.  Equivalence captures e.g. start_i == goal_j when
            # robots swap endpoints.
            equiv_next = (
                node_equiv.get(next_node, [next_node])
                if node_equiv else [next_node]
            )
            if any(
                reservation.get(en, {}).get(next_time, set())
                for en in equiv_next
            ):
                continue

            # Reject edge-swap conflicts: if another robot moves from
            # next_node → node at the same timestep, the two robots
            # would cross through each other during interpolation.
            # Use position equivalence so geometric swaps are caught
            # even when the graph nodes are technically different.
            if next_node != node and _swap_conflict_equiv(
                node, next_node, time, reservation, node_equiv,
            ):
                continue

            new_g = g + (0 if next_node == node else 1)
            h = dist_to_goal.get(next_node, T)
            new_f = new_g + h
            counter += 1
            heapq.heappush(pq, (new_f, new_g, counter, next_node, next_time))
            next_state = (next_node, next_time)
            if next_state not in came_from:
                came_from[next_state] = (node, time)

    return None


def _swap_conflict(
    from_node: str,
    to_node: str,
    time: int,
    reservation: Dict[str, Dict[int, Set[int]]],
) -> bool:
    """Check if moving from_node→to_node at ``time`` swaps with a reserved robot.

    Returns True if any robot currently reserved at ``to_node`` at ``time``
    is reserved at ``from_node`` at ``time + 1`` — i.e. moving in the
    opposite direction on the same edge at the same timestep.
    """
    robots_at_to = reservation.get(to_node, {}).get(time, set())
    if not robots_at_to:
        return False
    robots_at_from_next = reservation.get(from_node, {}).get(time + 1, set())
    # If any robot is at to_node now AND at from_node next step, it's a swap
    return bool(robots_at_to & robots_at_from_next)


def _swap_conflict_equiv(
    from_node: str,
    to_node: str,
    time: int,
    reservation: Dict[str, Dict[int, Set[int]]],
    node_equiv: Optional[Dict[str, List[str]]],
) -> bool:
    """Position-aware swap detection.

    A swap is detected if any robot reserved at a node geometrically
    equivalent to ``to_node`` at ``time`` is reserved at a node
    geometrically equivalent to ``from_node`` at ``time + 1``.
    """
    if node_equiv is None:
        return _swap_conflict(from_node, to_node, time, reservation)
    equiv_to = node_equiv.get(to_node, [to_node])
    equiv_from = node_equiv.get(from_node, [from_node])
    robots_at_to = set()
    for n in equiv_to:
        robots_at_to |= reservation.get(n, {}).get(time, set())
    if not robots_at_to:
        return False
    robots_at_from_next = set()
    for n in equiv_from:
        robots_at_from_next |= reservation.get(n, {}).get(time + 1, set())
    return bool(robots_at_to & robots_at_from_next)


def _capacity_ok(
    node: str,
    time: int,
    reservation: Dict[str, Dict[int, Set[int]]],
    node_to_cells: Dict[str, List[int]],
    densities: Dict[int, int],
    hlg: HighLevelGraph,
) -> bool:
    """Check whether placing one more robot at ``node`` at ``time`` respects
    all incident cell capacities."""
    for ci in node_to_cells.get(node, []):
        cap = densities.get(ci)
        if cap is None:
            continue
        # Count robots currently reserved at nodes incident to cell ci at time
        count = 0
        for n in hlg.cell_incident_nodes.get(ci, []):
            count += len(reservation.get(n, {}).get(time, set()))
        if count >= cap:
            return False
    return True


# ---------------------------------------------------------------------------
# Priority-based strategy
# ---------------------------------------------------------------------------

def _solve_priority(
    hlg: HighLevelGraph,
    num_robots: int,
    T: int,
    verbose: bool,
) -> Optional[MCFSolution]:
    """Sequential planning with longest-path-first priority ordering."""
    G = hlg.graph

    # Compute shortest path length per robot; plan longest first
    priorities: List[Tuple[int, int]] = []
    for r in range(num_robots):
        src = f"start_{r}"
        dst = f"goal_{r}"
        try:
            dist = nx.shortest_path_length(G, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = T  # unreachable → highest priority (plan first)
        priorities.append((dist, r))

    priorities.sort(reverse=True)
    order = [r for _, r in priorities]

    if verbose:
        print(f"Priority order: {order}")

    return _solve_sequential(hlg, num_robots, T, verbose, robot_order=order)
