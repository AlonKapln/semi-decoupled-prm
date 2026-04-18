"""Staged multi-robot motion planning solver.

Steps 7–8 of ``plan.tex``.  Orchestrates the full pipeline:

1. ``free_space_builder.construct_free_space`` — Minkowski-inflated
   arrangement (step 1).
2. ``scene_partitioning.partition_free_space_grid`` — grid decomposition
   of free space into convex-or-near-convex cells with per-cell density
   (steps 2–3).
3. ``high_level_graph.build_high_level_graph`` — cell-adjacency graph
   with boundary ports, robot starts/goals; also precomputes a
   per-cell inset for every port so path realisation can use it
   directly (step 5).
4. ``mcf_solver.solve_mcf`` — multi-commodity flow routing (step 6).
5. **Path realisation** — for each timestep, build a fresh ad-hoc
   joint PRM per active cell via ``cell_joint_prm.build_adhoc_roadmap``
   and extract a joint path entry→exit.
6. **Temporal stitching** — pad segments so all robots' ``Path``s share
   a common length.

Note: no pre-built per-cell PRM. The ad-hoc PRM at path realisation
does all joint sampling — the pre-build step would be redundant and
strictly worse (it couldn't know which robots coexist in a cell at
each timestep, so it couldn't pin holders or include correct transit
entries/exits).

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
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx

from CGALPY.CGALPY import Bounded_side
from discopygal.bindings import FT, Point_2
from discopygal.geometry_utils.collision_detection import ObjectCollisionDetection
from discopygal.solvers_infra import PathCollection, PathPoint, Path
from discopygal.solvers_infra.Solver import Solver

from cell_joint_prm import build_adhoc_roadmap
from high_level_graph import (
    HighLevelGraph,
    build_high_level_graph,
    estimate_time_horizon,
)
from mcf_solver import MCFSolution, solve_mcf
from partition import Partition
from scene_partitioning import partition_free_space_grid
from visualize_cells import draw_cells


@dataclass
class RealisationFailure:
    """Why a particular (cell, timestep) could not be realised.

    Surfaced by ``_realise_timestep`` and propagated by ``_realise_paths``
    so ``_solve`` can run Level B retry (lower the offending cell's
    density and re-run MCF).
    """

    cell_id: int
    timestep: int
    transit_robots: List[int]
    holders: List[int]
    reason: str  # from AdhocResult.reason


def _point_in_partition(partition: Partition, x: float, y: float) -> bool:
    """Closed point-in-polygon test against a Partition's CGAL polygon."""
    side = partition.polygon.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _node_position_in_cell(
    hlg: HighLevelGraph,
    cell_id: int,
    node_name: str,
) -> Optional[Tuple[float, float]]:
    """Position of an HLG node *as seen from a specific cell*.

    - ``port_N`` → the precomputed per-cell inset from
      ``hlg.cell_boundary_ports[cell_id][N]``.
    - ``start_*`` / ``goal_*`` → the raw position from
      ``hlg.node_positions`` (already inside the cell).
    - unknown / missing → ``None``.
    """
    if node_name.startswith("port_"):
        try:
            port_id = int(node_name[len("port_"):])
        except ValueError:
            return None
        return hlg.cell_boundary_ports.get(cell_id, {}).get(port_id)
    return hlg.node_positions.get(node_name)


class StagedSolver(Solver):
    """Staged multi-robot solver following the ``plan.tex`` pipeline.

    Parameters
    ----------
    num_samples : int
        PRM samples per cell.
    k_nearest : int
        PRM neighbour connections per sample.
    time_horizon : int or None
        MCF time horizon.  ``None`` = auto-compute.
    prm_seed : int or None
        RNG seed for reproducible PRM sampling.
    max_cell_density : int
        Upper bound for per-cell density used by the grid refinement. Lower
        values split large open regions into smaller cells (better joint-PRM
        behaviour, more expensive HLG/MCF); higher values leave cells as
        large as the free-space topology allows.
    """

    def __init__(
        self,
        num_samples: int = 50,
        k_nearest: int = 8,
        time_horizon: Optional[int] = None,
        prm_seed: Optional[int] = None,
        max_cell_density: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.time_horizon = time_horizon
        self.prm_seed = prm_seed
        self.max_cell_density = max_cell_density

        # Populated during solve for solver_viewer visualisation
        self._arrangement = None
        self._hlg = None

        # P1.1 sample-reuse cache: ``(cell_id, n_transit, frozenset of
        # rounded pinned positions)`` → list of interior joint configs
        # accepted by the most recent ``build_adhoc_roadmap`` call with
        # that shape. Reset per solve in ``_solve``.
        self._sample_cache: Dict[
            Tuple[int, int, frozenset], List[Tuple[Tuple[float, float], ...]]
        ] = {}

    def get_arrangement(self):
        """Return the free-space arrangement for solver_viewer display."""
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
            "num_samples": ("PRM samples per cell:", 50, int),
            "k_nearest": ("PRM k-nearest neighbours:", 8, int),
            "time_horizon": ("MCF time horizon (0=auto):", 0, int),
            "prm_seed": ("PRM RNG seed (0=random):", 0, int),
            "max_cell_density": ("Max cell density (grid refinement):", 100, int),
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
        # Fresh cache per solve — stale entries across solves would
        # feed the wrong cell into ``build_adhoc_roadmap``.
        self._sample_cache = {}

        def log(msg: str) -> None:
            if verbose and writer:
                print(msg, file=writer)
            elif verbose:
                print(msg)

        # --- Step 1+2+3: free space → cells with density ---
        log("Step 1-3: building free space and grid decomposition...")
        partitions, arrangement = partition_free_space_grid(
            scene, robot_radius, max_cell_density=self.max_cell_density,
        )
        self._arrangement = arrangement
        log(f"  {len(partitions)} cells")

        # Scene-level collision checker used by:
        # 1. ``build_high_level_graph`` to place each port at a point on
        #    the shared edge where an ``r``-inward inset is checker-valid
        #    (not inside an inflated-obstacle band), dropping unusable
        #    edges outright.
        # 2. The ad-hoc PRM to verify sampled configs and steered edges
        #    against the *true* inflated obstacle geometry (arcs included),
        #    complementing the polygonal containment test. The polygonal
        #    approximation of inflated-obstacle arcs is inexact, so the
        #    checker is essential — it overrides the polygon test for
        #    curved regions.
        if len(scene.obstacles) > 0:
            self._collision_checker = ObjectCollisionDetection(
                scene.obstacles, robots[0],
            )
        else:
            self._collision_checker = None

        # --- Step 5: high-level graph ---
        log("Step 5: building high-level graph...")
        robot_starts = [
            (r.start.x().to_double(), r.start.y().to_double()) for r in robots
        ]
        robot_goals = [
            (r.end.x().to_double(), r.end.y().to_double()) for r in robots
        ]

        hlg = build_high_level_graph(
            partitions, robot_starts, robot_goals, robot_radius,
            checker=self._collision_checker,
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

        # --- Save a cell decomposition snapshot ---
        # Every solve drops a PNG named ``{scene}_d{density}.png`` into
        # visualizations/ so the user can eyeball the partition + port
        # layout that MCF is about to plan on. The scene stem is read
        # from ``scene._source_path`` if the caller set it (test suite /
        # CLI do); otherwise we fall back to ``"scene"``. Failures here
        # (missing matplotlib, display issues) must never break solving.
        try:
            src = getattr(scene, "_source_path", None)
            stem = os.path.splitext(os.path.basename(src))[0] if src else "scene"
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "visualizations",
                f"{stem}_d{self.max_cell_density}.png",
            )
            title = (
                f"{stem}, density={self.max_cell_density}, "
                f"{len(partitions)} cells, "
                f"{sum(len(p) for p in hlg.cell_boundary_ports.values()) // 2} ports, "
                f"{num_robots} robots"
            )
            out = draw_cells(
                scene, partitions, hlg, robot_radius,
                save_path=save_path, title=title,
            )
            if out is not None:
                log(f"  cell snapshot → {out}")
        except Exception as exc:  # noqa: BLE001 — viz is best-effort
            log(f"  cell snapshot skipped: {exc}")

        # --- Steps 6-8: MCF + path realisation with Level B retry ---
        # P0.2 Level B: if path realisation fails for a specific cell, drop
        # that cell's density by 1, rebuild affected HLG edge capacities,
        # and re-run MCF + realisation. Cap at ``max_retries`` attempts.
        T = self.time_horizon or estimate_time_horizon(hlg)
        max_retries = 3

        for attempt in range(max_retries + 1):
            log(f"Step 6: solving MCF (T={T}, attempt={attempt})...")
            mcf_sol = solve_mcf(hlg, num_robots, T, verbose=verbose)
            if mcf_sol is None:
                log("  MCF found no solution")
                return PathCollection()

            log("Steps 7-8: realising geometric paths...")
            result = self._realise_paths(
                mcf_sol, hlg, partitions, robots, robot_radius, log,
            )
            if isinstance(result, PathCollection):
                return result

            failure = result
            part = partitions[failure.cell_id]
            if part.density <= 1:
                log(
                    f"  cell {failure.cell_id} already at density 1 and "
                    f"still failing (reason={failure.reason}, "
                    f"t={failure.timestep}, "
                    f"transits={failure.transit_robots}). Bailing."
                )
                return PathCollection()
            # Halve instead of decrement: cells that need to collapse
            # from e.g. density 8 to 1 would otherwise need 7 Level B
            # attempts (each rerunning MCF). Halving converges in
            # ⌈log2(d)⌉ retries.
            new_density = max(1, part.density // 2)
            log(
                f"  Level B retry: cell {failure.cell_id} density "
                f"{part.density} → {new_density} "
                f"(reason={failure.reason}, t={failure.timestep})"
            )
            part.update_density(new_density)
            # Keep HLG edge capacities in sync with the cell density —
            # MCF reads capacity from edge attributes.
            for u, v, data in hlg.graph.edges(data=True):
                if data.get("cell_id") == failure.cell_id:
                    data["capacity"] = new_density

        log(f"  Exhausted {max_retries} Level B retries. Bailing.")
        return PathCollection()

    # ------------------------------------------------------------------
    # Path realisation — ad-hoc per-(cell, timestep) joint PRM planning
    # ------------------------------------------------------------------

    def _realise_paths(
        self,
        mcf_sol: MCFSolution,
        hlg: HighLevelGraph,
        partitions: List[Partition],
        robots,
        robot_radius: float,
        log,
    ) -> Union[PathCollection, RealisationFailure]:
        """Convert MCF node sequences into geometric discopygal paths.

        For each timestep, every cell with ≥ 1 active robot gets a fresh
        ad-hoc joint PRM built via ``cell_joint_prm.build_adhoc_roadmap``.
        Transit robots contribute explicit entry/exit configurations
        (precomputed per-cell port insets from the HLG); holders are
        pinned at their actual current positions. On ad-hoc PRM failure
        the method propagates a :class:`RealisationFailure` up to
        ``_solve`` so the outer retry loop can lower the offending
        cell's density and try again.
        """
        num_robots = len(robots)
        T = len(mcf_sol[0])

        # Current geometric position per robot. Updated at end of every
        # timestep from the robot's segment exit. At a cell boundary
        # this will not equal the next timestep's computed entry — that
        # difference *is* the geometric cell crossing and is materialised
        # in the waypoint list.
        current_pos: Dict[int, Tuple[float, float]] = {
            r: (
                robots[r].start.x().to_double(),
                robots[r].start.y().to_double(),
            )
            for r in range(num_robots)
        }

        robot_segments: Dict[int, List[List[Tuple[float, float]]]] = {
            r: [] for r in range(num_robots)
        }

        seed_base = self.prm_seed if self.prm_seed is not None else 0

        for t in range(T - 1):
            timestep_result = self._realise_timestep(
                t, mcf_sol, hlg, partitions, current_pos, robot_radius,
                seed_base, log,
            )
            if isinstance(timestep_result, RealisationFailure):
                return timestep_result
            timestep_segments = timestep_result

            # ---- Pad segments to a common length for this timestep ----
            max_seg = max(
                (len(s) for s in timestep_segments.values()), default=1,
            )
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
            goal = (
                robots[r].end.x().to_double(),
                robots[r].end.y().to_double(),
            )
            # The last PRM exit config already places the robot at its
            # literal goal via ``_node_position_in_cell("goal_r")``.
            # Assert the invariant rather than silently forcing it.
            assert math.hypot(wp[-1][0] - goal[0], wp[-1][1] - goal[1]) < 1e-6, (
                f"robot {r}: last waypoint {wp[-1]} != declared goal {goal}"
            )
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

    def _realise_timestep(
        self,
        t: int,
        mcf_sol: MCFSolution,
        hlg: HighLevelGraph,
        partitions: List[Partition],
        current_pos: Dict[int, Tuple[float, float]],
        robot_radius: float,
        seed_base: int,
        log,
    ) -> Union[Dict[int, List[Tuple[float, float]]], RealisationFailure]:
        """Plan one MCF timestep.

        Returns per-robot segments on success, or a ``RealisationFailure``
        describing the cell / timestep / reason on failure. The outer
        ``_solve`` retry loop uses the failure to target a density reduction.
        """
        num_robots = len(current_pos)

        # Classify robots: holding (src == dst), or transiting via a
        # specific cell (identified by edge cell_id in the HLG).
        cell_transitions: Dict[int, List[Tuple[int, str, str]]] = defaultdict(
            list,
        )
        holding: List[int] = []

        for r in range(num_robots):
            src = mcf_sol[r][t]
            dst = mcf_sol[r][t + 1]
            if src == dst:
                holding.append(r)
                continue
            cell_id = hlg.graph.edges[src, dst].get("cell_id")
            cell_transitions[cell_id].append((r, src, dst))

        # Holders sit in their current cell for this timestep; we locate
        # them by point-in-polygon so their positions can be pinned in
        # the right ad-hoc PRM.
        cell_holding: Dict[int, List[int]] = defaultdict(list)
        for r in holding:
            px, py = current_pos[r]
            for ci, p in enumerate(partitions):
                if _point_in_partition(p, px, py):
                    cell_holding[ci].append(r)
                    break

        timestep_segments: Dict[int, List[Tuple[float, float]]] = {
            r: [current_pos[r]] for r in holding
        }

        for cell_id, transitions in cell_transitions.items():
            partition = partitions[cell_id]
            holders_here = cell_holding.get(cell_id, [])

            transit_entries: List[Tuple[float, float]] = []
            transit_exits: List[Tuple[float, float]] = []
            for _r, src, dst in transitions:
                entry = _node_position_in_cell(hlg, cell_id, src)
                ex = _node_position_in_cell(hlg, cell_id, dst)
                if entry is None or ex is None:
                    log(
                        f"  ad-hoc PRM: missing node position at t={t}, "
                        f"cell={cell_id}"
                    )
                    return None
                transit_entries.append(entry)
                transit_exits.append(ex)

            pinned = [current_pos[r] for r in holders_here]

            base_samples = self._effective_num_samples(partition)
            cfg_path, reason = self._plan_adhoc(
                partition, transit_entries, transit_exits, pinned,
                robot_radius, seed_base, cell_id, t,
                base_samples=base_samples,
            )
            if cfg_path is None:
                transit_robots = [rr for rr, _, _ in transitions]
                log(
                    f"  ad-hoc PRM failed at t={t}, cell={cell_id}, "
                    f"transits={transit_robots}, holders={holders_here}, "
                    f"reason={reason}"
                )
                return RealisationFailure(
                    cell_id=cell_id,
                    timestep=t,
                    transit_robots=transit_robots,
                    holders=list(holders_here),
                    reason=reason or "unknown",
                )

            n_transit = len(transitions)
            for slot, (r, _src, _dst) in enumerate(transitions):
                timestep_segments[r] = [
                    (cfg[slot][0], cfg[slot][1]) for cfg in cfg_path
                ]
            for j, r in enumerate(holders_here):
                slot = n_transit + j
                timestep_segments[r] = [
                    (cfg[slot][0], cfg[slot][1]) for cfg in cfg_path
                ]

        return timestep_segments

    # ------------------------------------------------------------------
    # Ad-hoc PRM helpers
    # ------------------------------------------------------------------

    def _effective_num_samples(self, partition: Partition) -> int:
        """Sample count for this cell's ad-hoc PRM.

        Scales ``self.num_samples`` by ``min(4.0, √complexity)`` so harder
        cells (elongated, arc-carved) get proportionally more samples. The
        square root dampens; the ``min(4.0, …)`` cap prevents pathological
        cells from blowing up runtime.
        """
        multiplier = min(4.0, math.sqrt(partition.complexity))
        return max(1, int(self.num_samples * multiplier))

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
        base_samples: Optional[int] = None,
    ) -> Tuple[Optional[List[Tuple[Tuple[float, float], ...]]], Optional[str]]:
        """Build an ad-hoc joint PRM and return ``(cfg_path, reason)``.

        Level A retry: escalate sample count exponentially (1×, 2×, 4×, 8×).
        If the first attempt fails with ``entry_infeasible`` or
        ``exit_infeasible`` we short-circuit — no amount of extra sampling
        rescues a joint endpoint that itself violates pairwise 2r or cell
        containment. Those failures go straight back to the caller so the
        outer solver retry (Level B) can decide what to do.

        ``base_samples`` overrides ``self.num_samples`` (used by P0.1 to
        scale per-cell by geometric complexity). ``None`` → use
        ``self.num_samples`` as-is.
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

        # P1.1 cache key: cell + transit count + pinned-holder set. Pinned
        # positions are rounded so tiny float drift between timesteps
        # (holders re-pinned from their last segment's final waypoint)
        # does not invalidate the cache.
        cache_key = (
            cell_id,
            len(transit_entries),
            frozenset(
                (round(x, 6), round(y, 6)) for x, y in pinned
            ),
        )
        cached = self._sample_cache.get(cache_key)

        base = base_samples if base_samples is not None else self.num_samples
        last_reason: Optional[str] = None
        for attempt, nsamples in enumerate((base, base * 2, base * 4, base * 8)):
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
                reuse_samples=cached,
            )
            last_reason = result.reason
            if result.reason in ("entry_infeasible", "exit_infeasible"):
                # More samples won't fix a bad endpoint.
                return None, result.reason
            if result.graph is None:
                continue
            entry_cfg, exit_cfg = result.entry_cfg, result.exit_cfg
            if entry_cfg == exit_cfg:
                self._sample_cache[cache_key] = list(result.interior_samples)
                return [entry_cfg], None
            try:
                path = nx.shortest_path(
                    result.graph, entry_cfg, exit_cfg, weight="weight",
                )
                self._sample_cache[cache_key] = list(result.interior_samples)
                return path, None
            except nx.NetworkXNoPath:
                last_reason = "no_path"
                continue
        return None, last_reason

