"""Staged multi-robot motion planning solver.

Steps 7–8 of ``plan.tex``.  Orchestrates the full pipeline:

1. ``free_space_builder.construct_free_space`` — Minkowski-inflated
   arrangement (step 1).
2. ``scene_partitioning.partition_free_space_vertical`` — trapezoidal
   decomposition into convex cells with per-cell density (steps 2–3).
3. ``high_level_graph.build_high_level_graph`` — cell-adjacency graph
   with boundary ports, robot starts/goals (step 5).
4. ``cell_joint_prm.build_cell_roadmap`` — joint k-robot PRM per cell
   (step 4).
5. ``high_level_graph.prune_by_prm`` — remove infeasible high-level
   edges.
6. ``mcf_solver.solve_mcf`` — multi-commodity flow routing (step 6).
7. **Path realisation** — for each robot, walk its MCF node sequence,
   query the cell PRM for a geometric path between consecutive port
   configs, and concatenate into a discopygal ``Path``.
8. **Temporal stitching** — insert hold ``PathPoint``s at ports so all
   robots are time-synchronised and the ``PathCollection`` is consistent.

Path realisation (step 7)
-------------------------
The MCF gives each robot a time-indexed sequence of high-level nodes::

    robot 0 : [start_0, port_2, port_5, goal_0, goal_0, ...]

Each consecutive *distinct* pair ``(port_a, port_b)`` lies on an edge
labelled with some cell ``c``.  In that cell's ``CellRoadmap`` the two
ports correspond to joint configurations.  We query the PRM for a
``networkx`` shortest path between them, which yields a sequence of
joint configurations.  From each joint config we extract the slot-0
position (or whichever slot is assigned to this robot) as a
``PathPoint``.

For the first and last segments (start → first port, last port → goal)
we connect the robot's exact start/goal position to the nearest PRM node
and route through the PRM.

Temporal stitching (step 8)
---------------------------
Robots may reach their cell-exit port at different PRM path lengths.
To keep the ``PathCollection`` synchronised (every ``Path`` must have
the same number of points for discopygal's interpolation) we pad shorter
segments with **hold points** — repeated ``PathPoint``s at the port
position.  The final paths are normalised to equal length.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from discopygal.bindings import FT, Point_2
from discopygal.solvers_infra import PathCollection, PathPoint, Path
from discopygal.solvers_infra.Solver import Solver

from cell_joint_prm import CellRoadmap, JointConfig, build_cell_roadmap
from high_level_graph import (
    HighLevelGraph,
    build_high_level_graph,
    estimate_time_horizon,
    prune_by_prm,
)
from mcf_solver import MCFSolution, solve_mcf
from scene_partitioning import partition_free_space_vertical


class StagedSolver(Solver):
    """Staged multi-robot solver following the ``plan.tex`` pipeline.

    Parameters
    ----------
    topology : str
        High-level graph topology: ``"pairwise"`` or ``"star"``.
    flow_strategy : str
        MCF solving strategy: ``"ilp"``, ``"sequential"``, or ``"priority"``.
    num_samples : int
        PRM samples per cell.
    k_nearest : int
        PRM neighbour connections per sample.
    time_horizon : int or None
        MCF time horizon.  ``None`` = auto-compute.
    prm_seed : int or None
        RNG seed for reproducible PRM sampling.
    """

    def __init__(
        self,
        topology: str = "pairwise",
        flow_strategy: str = "sequential",
        num_samples: int = 50,
        k_nearest: int = 8,
        time_horizon: Optional[int] = None,
        prm_seed: Optional[int] = None,
        prm_prune: str = "False",
        max_cell_density: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topology = topology
        self.flow_strategy = flow_strategy
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.time_horizon = time_horizon
        self.prm_seed = prm_seed
        self.prm_prune = eval(prm_prune)
        self.max_cell_density = max_cell_density

        # Populated during solve for solver_viewer visualisation
        self._arrangement = None
        self._hlg = None

    def get_arrangement(self):
        """Return the trapezoidal arrangement for solver_viewer display."""
        return self._arrangement

    def get_graph(self):
        """Return the high-level graph for solver_viewer display.

        Converts node positions to ``Point_2`` so solver_viewer can draw
        them.
        """
        if self._hlg is None:
            return None
        G = nx.Graph()
        for u, v in self._hlg.graph.edges():
            ux, uy = self._hlg.node_positions[u]
            vx, vy = self._hlg.node_positions[v]
            G.add_edge(Point_2(FT(ux), FT(uy)), Point_2(FT(vx), FT(vy)))
        return G

    @classmethod
    def get_arguments(cls):
        return {
            "topology": ("Graph topology (pairwise/star):", "pairwise", str),
            "flow_strategy": (
                "MCF strategy (ilp/sequential/priority):",
                "sequential",
                str,
            ),
            "num_samples": ("PRM samples per cell:", 50, int),
            "k_nearest": ("PRM k-nearest neighbours:", 8, int),
            "time_horizon": ("MCF time horizon (0=auto):", 0, int),
            "prm_seed": ("PRM RNG seed (0=random):", 0, int),
            "prm_prune": ("Prune infeasible HLG edges via PRM:", "False", str),
            "max_cell_density": ("Max cell density (grid refinement):", 4, int),
            **super().get_arguments(),
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _solve(self) -> PathCollection:
        scene = self.scene
        robots = scene.robots
        num_robots = len(robots)
        robot_radius = robots[0].radius.to_double()
        verbose = getattr(self, "verbose", False)
        writer = getattr(self, "writer", None)

        def log(msg: str) -> None:
            if verbose and writer:
                print(msg, file=writer)
            elif verbose:
                print(msg)

        # --- Step 1+2+3: free space → trapezoidal cells with density ---
        log("Step 1-3: building free space and trapezoidal decomposition...")
        partitions, arrangement = partition_free_space_vertical(
            scene, robot_radius, max_cell_density=self.max_cell_density,
        )
        self._arrangement = arrangement
        log(f"  {len(partitions)} cells")

        # --- Step 5: high-level graph ---
        log(f"Step 5: building high-level graph (topology={self.topology})...")
        robot_starts = [
            (r.start.x().to_double(), r.start.y().to_double()) for r in robots
        ]
        robot_goals = [
            (r.end.x().to_double(), r.end.y().to_double()) for r in robots
        ]

        hlg = build_high_level_graph(
            partitions, robot_starts, robot_goals, robot_radius,
            topology=self.topology,
        )
        self._hlg = hlg
        log(
            f"  V={hlg.graph.number_of_nodes()} "
            f"E={hlg.graph.number_of_edges()} "
            f"ports={sum(len(p) for p in hlg.cell_boundary_ports.values()) // 2}"
        )

        # Check all robots placed
        for r in range(num_robots):
            if r not in hlg.start_cells or r not in hlg.goal_cells:
                log(f"  Robot {r} start/goal outside free space!")
                return PathCollection()

        # --- Refine cell densities: cap by port count ---
        # A cell with n ports can have at most n robots transiting
        # simultaneously. Also ensure cells with starts/goals have
        # enough capacity for the robots originating/ending there.
        for ci, partition in enumerate(partitions):
            num_ports = len(hlg.cell_boundary_ports.get(ci, {}))
            num_starts = sum(
                1 for ri in range(num_robots) if hlg.start_cells.get(ri) == ci
            )
            num_goals = sum(
                1 for ri in range(num_robots) if hlg.goal_cells.get(ri) == ci
            )
            port_cap = max(num_ports, num_starts, num_goals, 1)
            new_density = min(partition.density, port_cap)
            if new_density != partition.density:
                log(f"  cell {ci}: density {partition.density} → {new_density} (ports={num_ports})")
                partition.update_density(new_density)

        # --- Step 4: per-cell PRM ---
        log("Step 4: building per-cell joint PRMs...")
        cell_roadmaps: List[CellRoadmap] = []
        for ci, partition in enumerate(partitions):
            ports = hlg.cell_boundary_ports.get(ci, {})
            rm = build_cell_roadmap(
                partition,
                boundary_ports=ports,
                num_samples=self.num_samples,
                k_nearest=self.k_nearest,
                seed=self.prm_seed or None,
            )
            cell_roadmaps.append(rm)
            log(
                f"  cell {ci}: k={partition.density} "
                f"V={rm.graph.number_of_nodes()} "
                f"E={rm.graph.number_of_edges()} "
                f"ports={len(rm.port_nodes)}/{len(ports)}"
            )

        # --- PRM pruning (optional) ---
        if self.prm_prune:
            pruned = prune_by_prm(hlg, cell_roadmaps)
            if pruned:
                log(f"  pruned {pruned} infeasible high-level edges")

        # --- Step 6: MCF ---
        T = self.time_horizon or estimate_time_horizon(hlg)
        log(f"Step 6: solving MCF (strategy={self.flow_strategy}, T={T})...")

        mcf_sol = solve_mcf(
            hlg, num_robots, T,
            strategy=self.flow_strategy, verbose=verbose,
        )
        if mcf_sol is None:
            log("  MCF found no solution")
            return PathCollection()

        # --- Steps 7+8: path realisation & stitching ---
        log("Steps 7-8: realising geometric paths...")
        return self._realise_paths(
            mcf_sol, hlg, cell_roadmaps, robots, robot_radius, log,
        )

    # ------------------------------------------------------------------
    # Path realisation (timestep-aware joint PRM planning)
    # ------------------------------------------------------------------

    def _realise_paths(
        self,
        mcf_sol: MCFSolution,
        hlg: HighLevelGraph,
        cell_roadmaps: List[CellRoadmap],
        robots,
        robot_radius: float,
        log,
    ) -> PathCollection:
        """Convert MCF node sequences into geometric discopygal paths.

        Timestep-aware: robots transiting through the same cell at the
        same timestep share a single joint PRM path.  Each robot extracts
        its assigned slot, guaranteeing pairwise separation via the PRM's
        joint-space steer check.
        """
        num_robots = len(robots)
        T = len(mcf_sol[0])

        # Current geometric position per robot
        current_pos: Dict[int, Tuple[float, float]] = {}
        for r in range(num_robots):
            current_pos[r] = (
                robots[r].start.x().to_double(),
                robots[r].start.y().to_double(),
            )

        # Per-timestep segments for each robot
        robot_segments: Dict[int, List[List[Tuple[float, float]]]] = {
            r: [] for r in range(num_robots)
        }

        for t in range(T - 1):
            # --- Classify robots at this timestep ---
            # cell_id → [(robot_idx, src_node, dst_node)]
            cell_transitions: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)
            holding: List[int] = []
            fallback: List[Tuple[int, str, str]] = []

            for r in range(num_robots):
                src = mcf_sol[r][t]
                dst = mcf_sol[r][t + 1]
                if src == dst:
                    holding.append(r)
                else:
                    cell_id = self._edge_cell_id(hlg, src, dst)
                    if cell_id is not None and cell_id < len(cell_roadmaps):
                        cell_transitions[cell_id].append((r, src, dst))
                    else:
                        fallback.append((r, src, dst))

            timestep_segments: Dict[int, List[Tuple[float, float]]] = {}

            # Holding robots: stay at current position
            for r in holding:
                timestep_segments[r] = [current_pos[r]]

            # Transitions without a valid cell: straight line
            for r, src, dst in fallback:
                dst_pos = hlg.node_positions.get(dst, current_pos[r])
                timestep_segments[r] = [current_pos[r], dst_pos]

            # --- Joint PRM planning per cell ---
            for cell_id, transitions in cell_transitions.items():
                rm = cell_roadmaps[cell_id]
                k = 0
                if rm.graph.number_of_nodes() > 0:
                    k = len(next(iter(rm.graph.nodes)))

                if k == 0:
                    # Empty PRM: straight-line fallback
                    for r, src, dst in transitions:
                        dst_pos = hlg.node_positions.get(dst, current_pos[r])
                        timestep_segments[r] = [current_pos[r], dst_pos]
                    continue

                # Assign a unique slot to each transiting robot
                robot_slots: Dict[int, int] = {}
                entry_map: Dict[int, Tuple[float, float]] = {}
                exit_map: Dict[int, Tuple[float, float]] = {}

                for idx, (r, src, dst) in enumerate(transitions):
                    slot = min(idx, k - 1)
                    robot_slots[r] = slot
                    entry_map[slot] = current_pos[r]
                    exit_map[slot] = hlg.node_positions.get(
                        dst, current_pos[r],
                    )

                # Find a single joint PRM path for all robots in this cell
                joint_path = self._find_joint_prm_path(
                    rm.graph, entry_map, exit_map,
                )

                if joint_path:
                    for r, src, dst in transitions:
                        slot = robot_slots[r]
                        s = min(slot, len(joint_path[0]) - 1)
                        prm_seg = [(cfg[s][0], cfg[s][1])
                                   for cfg in joint_path]
                        # No bookend with raw port positions — PRM nodes
                        # are already nudged r inward from cell boundaries
                        # so the path stays r-inset for cross-cell safety.
                        timestep_segments[r] = [current_pos[r]] + prm_seg
                else:
                    for r, src, dst in transitions:
                        dst_pos = hlg.node_positions.get(
                            dst, current_pos[r],
                        )
                        timestep_segments[r] = [current_pos[r], dst_pos]

            # --- Normalise segment lengths for this timestep ---
            if timestep_segments:
                max_seg = max(len(s) for s in timestep_segments.values())
            else:
                max_seg = 1

            for r in range(num_robots):
                seg = timestep_segments.get(r, [current_pos[r]])
                while len(seg) < max_seg:
                    seg.append(seg[-1])
                robot_segments[r].append(seg)
                current_pos[r] = seg[-1]

        # --- Flatten segments into waypoints ---
        robot_waypoints: Dict[int, List[Tuple[float, float]]] = {}
        for r in range(num_robots):
            start = (
                robots[r].start.x().to_double(),
                robots[r].start.y().to_double(),
            )
            wp: List[Tuple[float, float]] = [start]
            for seg in robot_segments[r]:
                # Skip first point of segment (same as end of previous)
                wp.extend(seg[1:])

            # Ensure exact goal at the end
            goal = (
                robots[r].end.x().to_double(),
                robots[r].end.y().to_double(),
            )
            wp[-1] = goal
            robot_waypoints[r] = wp

        # Final pad to equal length
        max_len = max(
            (len(wp) for wp in robot_waypoints.values()), default=1,
        )
        for r in robot_waypoints:
            wp = robot_waypoints[r]
            while len(wp) < max_len:
                wp.append(wp[-1])

        # Build PathCollection
        pc = PathCollection()
        for r, robot in enumerate(robots):
            points = [
                PathPoint(Point_2(FT(x), FT(y)))
                for x, y in robot_waypoints[r]
            ]
            pc.add_robot_path(robot, Path(points))

        log(f"  {num_robots} paths, {max_len} waypoints each")
        return pc

    # ------------------------------------------------------------------
    # Joint PRM helpers
    # ------------------------------------------------------------------

    def _find_joint_prm_path(
        self,
        G: nx.Graph,
        entry_positions: Dict[int, Tuple[float, float]],
        exit_positions: Dict[int, Tuple[float, float]],
    ) -> Optional[List[JointConfig]]:
        """Find a PRM path matching entry and exit slot positions.

        Finds the PRM nodes whose slots best match the given entry/exit
        positions and returns the shortest weighted path between them.
        """
        if G.number_of_nodes() == 0:
            return None

        src = self._nearest_joint_node(G, entry_positions)
        dst = self._nearest_joint_node(G, exit_positions)

        if src is None or dst is None:
            return None
        if src == dst:
            return [src]

        try:
            return nx.shortest_path(G, src, dst, weight="weight")
        except nx.NetworkXNoPath:
            return None

    @staticmethod
    def _nearest_joint_node(
        G: nx.Graph,
        slot_positions: Dict[int, Tuple[float, float]],
    ) -> Optional[JointConfig]:
        """Find the PRM node minimizing total squared distance across slots.

        ``slot_positions`` maps slot indices to target ``(x, y)`` positions.
        The PRM node whose joint configuration best matches all specified
        slots simultaneously is returned.
        """
        if G.number_of_nodes() == 0:
            return None
        best: Optional[JointConfig] = None
        best_cost = float("inf")
        for cfg in G.nodes:
            cost = 0.0
            for slot, (px, py) in slot_positions.items():
                s = min(slot, len(cfg) - 1)
                dx = cfg[s][0] - px
                dy = cfg[s][1] - py
                cost += dx * dx + dy * dy
            if cost < best_cost:
                best_cost = cost
                best = cfg
        return best

    def _edge_cell_id(
        self, hlg: HighLevelGraph, src: str, dst: str,
    ) -> Optional[int]:
        """Find the cell_id of the edge between src and dst."""
        G = hlg.graph
        if G.has_edge(src, dst):
            return G.edges[src, dst].get("cell_id")
        # For star topology, try two-hop through hub
        for nb in G.neighbors(src):
            if nb.startswith("hub_") and G.has_edge(nb, dst):
                return G.edges[src, nb].get("cell_id")
        return None
