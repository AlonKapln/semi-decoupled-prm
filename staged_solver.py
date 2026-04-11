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

Path realisation (step 7) — ad-hoc per-(cell, timestep) joint PRM
------------------------------------------------------------------
The MCF gives each robot a time-indexed sequence of high-level nodes::

    robot 0 : [start_0, port_2, port_5, goal_0, goal_0, ...]

For each timestep ``t`` we group active robots by cell: a robot is
either **transiting** through some cell ``c`` (its edge ``src → dst``
is labelled with ``c``) or **holding** inside a cell (``src == dst``).
For every cell with ≥ 1 active robot we build a small **ad-hoc joint
PRM** on the fly via ``cell_joint_prm.build_adhoc_roadmap``:

- transit robots contribute an *entry* (inset-into-c position of
  ``src``) and *exit* (inset-into-c position of ``dst``),
- holding robots contribute a *pinned* position at their actual
  current location — they do not move during the timestep.

The ad-hoc PRM inserts the joint entry and exit as **explicit nodes**
(no "nearest neighbour" approximation), validates pairwise 2r
separation, and draws ``num_samples`` interior joint configurations
with holders pinned. A shortest path between entry and exit yields a
sequence of joint configurations; slot ``i`` of each config is the
position of the ``i``-th robot in the cell.

This kills the class of bugs caused by the pre-built per-cell joint
PRM: no holder teleportation, no slot-0-only port pinning, no
``NetworkXNoPath`` fallback to raw port midpoints.

**Cell boundary crossings.** Entries and exits are inset into the
*current* cell, so the robot's position at the end of timestep ``t``
(inside cell A) differs from the start of timestep ``t+1`` (inside
cell B). Discopygal's linear interpolation between the two consecutive
waypoints crosses the shared edge, performing the cell-crossing
geometrically. No raw port midpoint ever appears in a waypoint.

**Fail loud.** If an ad-hoc PRM is infeasible (entry/exit joint config
invalid, or no path found even after one retry with doubled samples),
the solver returns an empty ``PathCollection`` and logs which cell and
robots triggered it, instead of silently falling back to straight lines.

Temporal stitching (step 8)
---------------------------
Robots finishing their per-timestep joint PRM segment in different
numbers of waypoints are right-padded with a repeat of their final
position (a hold), so every robot's ``Path`` has the same length.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2
from discopygal.geometry_utils.collision_detection import ObjectCollisionDetection
from discopygal.solvers_infra import PathCollection, PathPoint, Path
from discopygal.solvers_infra.Solver import Solver

from cell_joint_prm import (
    CellRoadmap,
    build_adhoc_roadmap,
    build_cell_roadmap,
)
from high_level_graph import (
    HighLevelGraph,
    build_high_level_graph,
    estimate_time_horizon,
    prune_by_prm,
)
from mcf_solver import MCFSolution, solve_mcf
from partition import Partition
from scene_partitioning import (
    partition_free_space_grid,
    partition_free_space_vertical,
)


def _point_in_partition(partition: Partition, x: float, y: float) -> bool:
    """Closed point-in-polygon test against a Partition's CGAL polygon."""
    side = partition.polygon.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _polygon_vertices(partition: Partition) -> List[Tuple[float, float]]:
    return [
        (v.x().to_double(), v.y().to_double())
        for v in partition.polygon.vertices()
    ]


def _inset_toward_centroid(
    partition: Partition,
    position: Tuple[float, float],
    robot_radius: float,
) -> Tuple[float, float]:
    """Nudge a boundary position ``r`` into the cell interior.

    Robust to non-convex partitions: finds the polygon edge closest to
    ``position`` and steps ``r`` along its inward normal, verified by a
    point-in-polygon test. Falls back to the vertex-average direction
    only as a last resort.
    """
    poly = partition.polygon
    verts = _polygon_vertices(partition)
    n = len(verts)
    px, py = position
    if n == 0:
        return (px, py)

    # Locate the polygon edge closest to `position` (the "owning" edge).
    best_idx = -1
    best_d_sq = float("inf")
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        lensq = dx * dx + dy * dy
        if lensq <= 1e-24:
            continue
        s = ((px - ax) * dx + (py - ay) * dy) / lensq
        s = max(0.0, min(1.0, s))
        cx = ax + s * dx
        cy = ay + s * dy
        d_sq = (px - cx) ** 2 + (py - cy) ** 2
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best_idx = i

    if best_idx >= 0:
        ax, ay = verts[best_idx]
        bx, by = verts[(best_idx + 1) % n]
        dx, dy = bx - ax, by - ay
        edge_len = math.sqrt(dx * dx + dy * dy)
        if edge_len > 1e-12:
            nx_ = -dy / edge_len
            ny_ = dx / edge_len
            # Probe a small step along each perpendicular candidate.
            eps = max(robot_radius * 0.25, 1e-6)
            cand_plus = (px + eps * nx_, py + eps * ny_)
            cand_minus = (px - eps * nx_, py - eps * ny_)
            plus_inside = _point_in_partition(partition, *cand_plus)
            minus_inside = _point_in_partition(partition, *cand_minus)
            if plus_inside and not minus_inside:
                sign = 1.0
            elif minus_inside and not plus_inside:
                sign = -1.0
            elif plus_inside and minus_inside:
                # Both sides "inside" (non-convex) — pick the side whose
                # full r-step stays inside, preferring plus.
                full_plus = (px + robot_radius * nx_,
                             py + robot_radius * ny_)
                full_minus = (px - robot_radius * nx_,
                              py - robot_radius * ny_)
                if _point_in_partition(partition, *full_plus):
                    sign = 1.0
                elif _point_in_partition(partition, *full_minus):
                    sign = -1.0
                else:
                    sign = 1.0
            else:
                sign = None

            if sign is not None:
                # Shrink until the full step is inside (for tiny cells).
                step = robot_radius
                for _ in range(6):
                    cand = (px + sign * step * nx_,
                            py + sign * step * ny_)
                    if _point_in_partition(partition, *cand):
                        return cand
                    step *= 0.5
                return (px + sign * step * nx_,
                        py + sign * step * ny_)

    # Fallback: vertex-average direction.
    cx_avg = sum(x for x, _ in verts) / n
    cy_avg = sum(y for _, y in verts) / n
    dx, dy = cx_avg - px, cy_avg - py
    dist = math.sqrt(dx * dx + dy * dy)
    if dist <= 1e-12:
        return (px, py)
    nudge = min(robot_radius, max(dist - 1e-6, 0.0))
    return (px + (nudge / dist) * dx, py + (nudge / dist) * dy)


def _inset_node_in_cell(
    hlg: HighLevelGraph,
    partition: Partition,
    node_name: str,
    robot_radius: float,
) -> Optional[Tuple[float, float]]:
    """Return the position used for an HLG node when entering/exiting a cell.

    - ``port_*`` → inset the raw midpoint r toward the cell centroid so
      the position is inside the cell and ≥ r from the shared edge.
    - ``start_*`` / ``goal_*`` / ``hub_*`` → return the raw position as-is
      (these are already inside the cell).
    - unknown / missing → ``None``.
    """
    pos = hlg.node_positions.get(node_name)
    if pos is None:
        return None
    if node_name.startswith("port_"):
        return _inset_toward_centroid(partition, pos, robot_radius)
    return pos


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
    partition_mode : str
        ``"vertical"`` (default) — trapezoidal decomposition + grid refinement
        via ``partition_free_space_vertical``. Produces small convex cells,
        best for scenes with diagonal obstacles. ``"grid"`` — pure grid on the
        raw free-space arrangement via ``partition_free_space_grid``. Produces
        coarser, possibly non-convex cells with no narrow slivers, best for
        warehouse/corridor scenes where vertical decomposition creates cells
        narrower than 2r.
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
        partition_mode: str = "vertical",
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
        if partition_mode not in ("vertical", "grid"):
            raise ValueError(
                f"partition_mode must be 'vertical' or 'grid', got {partition_mode!r}"
            )
        self.partition_mode = partition_mode

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
            "max_cell_density": ("Max cell density (grid refinement):", 100, int),
            "partition_mode": (
                "Partition mode (vertical/grid):", "grid", str,
            ),
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

        # --- Step 1+2+3: free space → cells with density ---
        log(
            f"Step 1-3: building free space and decomposition "
            f"(partition_mode={self.partition_mode})..."
        )
        if self.partition_mode == "grid":
            partitions, arrangement = partition_free_space_grid(
                scene, robot_radius, max_cell_density=self.max_cell_density,
            )
        else:
            partitions, arrangement = partition_free_space_vertical(
                scene, robot_radius, max_cell_density=self.max_cell_density,
            )
        self._arrangement = arrangement
        log(f"  {len(partitions)} cells")

        # Scene-level collision checker used by the ad-hoc PRM to verify
        # sampled configs and steered edges against the *true* inflated
        # obstacle geometry (arcs included), complementing the polygonal
        # containment test. Only enabled in grid mode since vertical-mode
        # cells are convex and the polygonal approximation is exact there.
        if self.partition_mode == "grid" and len(scene.obstacles) > 0:
            self._collision_checker = ObjectCollisionDetection(
                scene.obstacles, robots[0],
            )
        else:
            self._collision_checker = None

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
    # Path realisation — ad-hoc per-(cell, timestep) joint PRM planning
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

        For each timestep, every cell with ≥ 1 active robot gets a fresh
        ad-hoc joint PRM built via ``cell_joint_prm.build_adhoc_roadmap``.
        Transit robots contribute explicit entry/exit configurations
        (r-inset port positions); holders are pinned at their actual
        current positions. A failure of the ad-hoc PRM is fatal — we
        return an empty ``PathCollection`` rather than contaminate the
        solution with straight-line fallbacks through raw port midpoints.
        """
        num_robots = len(robots)
        T = len(mcf_sol[0])
        partitions = [rm.partition for rm in cell_roadmaps]

        # Current geometric position per robot. Updated at end of every
        # timestep from the robot's segment exit. At a cell boundary
        # this will not equal the next timestep's computed entry — that
        # difference *is* the geometric cell crossing and is materialised
        # in the waypoint list.
        current_pos: Dict[int, Tuple[float, float]] = {}
        for r in range(num_robots):
            current_pos[r] = (
                robots[r].start.x().to_double(),
                robots[r].start.y().to_double(),
            )

        robot_segments: Dict[int, List[List[Tuple[float, float]]]] = {
            r: [] for r in range(num_robots)
        }

        seed_base = self.prm_seed if self.prm_seed is not None else 0

        for t in range(T - 1):
            # ---- Classify every robot at this timestep ----
            cell_transitions: Dict[
                int, List[Tuple[int, str, str]]
            ] = defaultdict(list)
            cell_holding: Dict[int, List[int]] = defaultdict(list)
            holding: List[int] = []
            fallback_transits: List[Tuple[int, str, str]] = []

            for r in range(num_robots):
                src = mcf_sol[r][t]
                dst = mcf_sol[r][t + 1]
                if src == dst:
                    holding.append(r)
                    continue
                cell_id = self._edge_cell_id(hlg, src, dst)
                if cell_id is not None and cell_id < len(cell_roadmaps):
                    cell_transitions[cell_id].append((r, src, dst))
                else:
                    fallback_transits.append((r, src, dst))

            # Point-in-polygon classification for holders so their
            # positions can be pinned in the right cell's ad-hoc PRM.
            for r in holding:
                px, py = current_pos[r]
                for ci, p in enumerate(partitions):
                    if _point_in_partition(p, px, py):
                        cell_holding[ci].append(r)
                        break

            timestep_segments: Dict[int, List[Tuple[float, float]]] = {}

            # Holders: no motion this timestep.
            for r in holding:
                timestep_segments[r] = [current_pos[r]]

            # Fallback transits (edge with no cell_id — should be rare,
            # typically star-topology quirks). Straight line to an
            # r-inset version of the destination so we never emit a raw
            # port midpoint as a waypoint.
            for r, _src, dst in fallback_transits:
                dst_pos = self._fallback_dst_position(
                    hlg, partitions, dst, robot_radius,
                )
                if dst_pos is None:
                    dst_pos = current_pos[r]
                timestep_segments[r] = [current_pos[r], dst_pos]

            # ---- Ad-hoc joint PRM per cell with active transits ----
            for cell_id, transitions in cell_transitions.items():
                partition = partitions[cell_id]
                holders_here = cell_holding.get(cell_id, [])

                transit_entries: List[Tuple[float, float]] = []
                transit_exits: List[Tuple[float, float]] = []
                missing = False
                for _r, src, dst in transitions:
                    entry = _inset_node_in_cell(
                        hlg, partition, src, robot_radius,
                    )
                    ex = _inset_node_in_cell(
                        hlg, partition, dst, robot_radius,
                    )
                    if entry is None or ex is None:
                        missing = True
                        break
                    transit_entries.append(entry)
                    transit_exits.append(ex)

                if missing:
                    log(
                        f"  ad-hoc PRM: missing node position at t={t}, "
                        f"cell={cell_id}"
                    )
                    return PathCollection()

                pinned = [current_pos[r] for r in holders_here]

                cfg_path = self._plan_adhoc(
                    partition,
                    transit_entries,
                    transit_exits,
                    pinned,
                    robot_radius,
                    seed_base,
                    cell_id,
                    t,
                )
                if cfg_path is None:
                    transit_robots = [rr for rr, _, _ in transitions]
                    log(
                        f"  ad-hoc PRM failed at t={t}, cell={cell_id}, "
                        f"transits={transit_robots}, holders={holders_here}"
                    )
                    return PathCollection()

                n_transit = len(transitions)
                for slot, (r, _src, _dst) in enumerate(transitions):
                    seg = [(cfg[slot][0], cfg[slot][1]) for cfg in cfg_path]
                    timestep_segments[r] = seg
                for j, r in enumerate(holders_here):
                    slot = n_transit + j
                    seg = [(cfg[slot][0], cfg[slot][1]) for cfg in cfg_path]
                    timestep_segments[r] = seg

            # ---- Pad segments to a common length for this timestep ----
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

        # ---- Flatten per-timestep segments into robot waypoints ----
        # Each segment is appended in full. Adjacent segments whose
        # boundary points happen to match (same cell continuation,
        # holds) produce consecutive duplicates; this is intentional —
        # it keeps every robot's flattened waypoint count in lock-step
        # with every other robot's, which is what discopygal's uniform
        # interpolation parameter needs for correct temporal sync.
        robot_waypoints: Dict[int, List[Tuple[float, float]]] = {}
        for r in range(num_robots):
            wp: List[Tuple[float, float]] = []
            for seg in robot_segments[r]:
                wp.extend(seg)
            if not wp:
                wp = [
                    (
                        robots[r].start.x().to_double(),
                        robots[r].start.y().to_double(),
                    )
                ]
            # Ensure the literal goal is the last waypoint (only legal
            # non-inset position at the end of a path).
            goal = (
                robots[r].end.x().to_double(),
                robots[r].end.y().to_double(),
            )
            wp[-1] = goal
            robot_waypoints[r] = wp

        max_len = max(
            (len(wp) for wp in robot_waypoints.values()), default=1,
        )
        for r in robot_waypoints:
            wp = robot_waypoints[r]
            while len(wp) < max_len:
                wp.append(wp[-1])

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
    # Ad-hoc PRM helpers
    # ------------------------------------------------------------------

    def _plan_adhoc(
        self,
        partition: Partition,
        transit_entries: List[Tuple[float, float]],
        transit_exits: List[Tuple[float, float]],
        pinned: List[Tuple[float, float]],
        robot_radius: float,
        seed_base: int,
        cell_id: int,
        t: int,
    ) -> Optional[List[Tuple[Tuple[float, float], ...]]]:
        """Build an ad-hoc joint PRM and return a node path entry→exit.

        Retries once with ``2 * num_samples`` if the first attempt fails
        to produce a path. Returns ``None`` on hard failure.
        """
        seed = hash(
            (
                seed_base,
                cell_id,
                t,
                tuple(transit_entries),
                tuple(transit_exits),
                tuple(pinned),
            )
        ) & 0xFFFFFFFF

        for attempt, nsamples in (
            (0, self.num_samples),
            (1, self.num_samples * 2),
        ):
            rng = random.Random((seed + attempt) & 0xFFFFFFFF)
            result = build_adhoc_roadmap(
                partition,
                entry_positions=transit_entries,
                exit_positions=transit_exits,
                pinned_positions=pinned,
                robot_radius=robot_radius,
                num_samples=nsamples,
                k_nearest=self.k_nearest,
                rng=rng,
                checker=self._collision_checker,
            )
            if result is None:
                continue
            graph, entry_cfg, exit_cfg = result
            if entry_cfg == exit_cfg:
                return [entry_cfg]
            try:
                return nx.shortest_path(
                    graph, entry_cfg, exit_cfg, weight="weight",
                )
            except nx.NetworkXNoPath:
                continue
        return None

    def _fallback_dst_position(
        self,
        hlg: HighLevelGraph,
        partitions: List[Partition],
        dst_node: str,
        robot_radius: float,
    ) -> Optional[Tuple[float, float]]:
        """Best-effort r-inset target for a fallback (no cell_id) transit.

        If ``dst_node`` is a port, inset into any cell that owns it;
        otherwise return the raw position (start/goal/hub, already
        inside a cell).
        """
        pos = hlg.node_positions.get(dst_node)
        if pos is None:
            return None
        if not dst_node.startswith("port_"):
            return pos
        try:
            port_id = int(dst_node[len("port_"):])
        except ValueError:
            return pos
        for ci, ports in hlg.cell_boundary_ports.items():
            if port_id in ports and 0 <= ci < len(partitions):
                return _inset_toward_centroid(
                    partitions[ci], pos, robot_radius,
                )
        return pos

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
