"""Staged multi-robot motion planning solver.

Pipeline: Minkowski free space -> grid partition -> high-level
cell-adjacency graph -> prioritized space-time A* routing -> per
(cell, timestep) joint PRM realisation -> temporal stitching.
"""

import hashlib
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
from partition import Partition
from prioritized_solver import RoutingSolution, solve_prioritized
from scene_partitioning import partition_free_space_grid
from visualize_cells import draw_cells


@dataclass
class RealisationFailure:
    cell_id: int
    timestep: int
    transit_robots: List[int]
    holders: List[int]
    reason: str


class RealisationError(Exception):
    def __init__(self, failure: "RealisationFailure"):
        super().__init__(
            f"realisation failed at cell={failure.cell_id}, "
            f"t={failure.timestep}, reason={failure.reason}"
        )
        self.failure = failure


def _infer_scene_stem(scene) -> str:
    """Stable visualisation filename stem: source-path basename if set,
    otherwise a short content hash of robots+obstacles."""
    src = getattr(scene, "_source_path", None)
    if src:
        return os.path.splitext(os.path.basename(src))[0]
    parts: List[str] = []
    for r in scene.robots:
        parts.append(
            f"r{r.radius.to_double():.4f}:"
            f"{r.start.x().to_double():.4f},{r.start.y().to_double():.4f}"
            f"->{r.end.x().to_double():.4f},{r.end.y().to_double():.4f}"
        )
    for o in scene.obstacles:
        poly = getattr(o, "poly", None)
        if poly is None:
            continue
        coords = [
            (v.x().to_double(), v.y().to_double()) for v in poly.vertices()
        ]
        parts.append("o:" + ";".join(f"{x:.4f},{y:.4f}" for x, y in coords))
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]
    return f"scene_{digest}"


def _point_in_partition(partition: Partition, x: float, y: float) -> bool:
    side = partition.polygon.bounded_side(Point_2(FT(x), FT(y)))
    return side != Bounded_side.ON_UNBOUNDED_SIDE


def _node_position_in_cell(
    hlg: HighLevelGraph,
    cell_id: int,
    node_name: str,
) -> Optional[Tuple[float, float]]:
    """Port nodes return the per-cell port point; start/goal return the raw position."""
    if node_name.startswith("port_"):
        try:
            port_id = int(node_name[len("port_"):])
        except ValueError:
            return None
        return hlg.cell_boundary_ports.get(cell_id, {}).get(port_id)
    return hlg.node_positions.get(node_name)


class StagedSolver(Solver):
    """Staged multi-robot solver.

    num_samples     PRM samples per cell.
    k_nearest       PRM neighbour connections per sample.
    time_horizon    Router time horizon; None = auto-compute.
    prm_seed        RNG seed for reproducible PRM sampling.
    max_cell_density
                    Upper bound on per-cell density used by grid refinement.
                    Lower values split large regions into more cells.
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

        self._arrangement = None
        self._hlg = None

        # (cell_id, n_transit, frozenset of rounded pinned positions) ->
        # interior joint configs accepted by the most recent adhoc roadmap
        # call with that shape. Reset per solve.
        self._sample_cache: Dict[
            Tuple[int, int, frozenset], List[Tuple[Tuple[float, float], ...]]
        ] = {}

    def get_arrangement(self):
        return self._arrangement

    def get_graph(self):
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
            "time_horizon": ("Router time horizon (0=auto):", 0, int),
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
        robot_radius = robots[0].radius.to_double()  # homogeneous radii
        verbose = getattr(self, "verbose", False)
        writer = getattr(self, "writer", None)
        self._sample_cache = {}

        def log(msg: str) -> None:
            if verbose and writer:
                print(msg, file=writer)
            elif verbose:
                print(msg)

        log("Step 1-3: building free space and grid decomposition...")
        partitions, arrangement = partition_free_space_grid(
            scene, robot_radius, max_cell_density=self.max_cell_density,
        )
        self._arrangement = arrangement
        log(f"  {len(partitions)} cells")

        # Arc-exact obstacle checker. The polygonal cell approximation
        # chord-approximates inflated-obstacle arcs, so for curved faces
        # the checker overrides the polygon containment test.
        if len(scene.obstacles) > 0:
            self._collision_checker = ObjectCollisionDetection(
                scene.obstacles, robots[0],
            )
        else:
            self._collision_checker = None

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

        for r in range(num_robots):
            if r not in hlg.start_cells or r not in hlg.goal_cells:
                log(f"  Robot {r} start/goal outside free space!")
                return PathCollection()

        # Cap each cell's density by (port_count, start_count, goal_count, 1).
        # A cell with n ports admits at most n simultaneous transits; cells
        # containing starts/goals need room for those robots.
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
                log(f"  cell {ci}: density {partition.density} -> {new_density} (ports={num_ports})")
                partition.update_density(new_density)

        # Best-effort snapshot of the decomposition.
        try:
            stem = _infer_scene_stem(scene)
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
                log(f"  cell snapshot -> {out}")
        except Exception as exc:
            log(f"  cell snapshot skipped: {exc}")

        # Routing + realisation with density-retry on realisation failure.
        # On failure we halve the offending cell's density (halving rather
        # than decrementing converges in ceil(log2(d)) attempts) and re-run
        # the router.
        T = self.time_horizon or estimate_time_horizon(hlg)
        max_retries = 3

        for attempt in range(max_retries + 1):
            log(f"Step 6: prioritized routing (T={T}, attempt={attempt})...")
            routing_sol = solve_prioritized(
                hlg, num_robots, T, verbose=verbose,
            )
            if routing_sol is None:
                log("  Prioritized router found no solution")
                return PathCollection()

            log("Steps 7-8: realising geometric paths...")
            try:
                return self._realise_paths(
                    routing_sol, hlg, partitions, robots, robot_radius, log,
                )
            except RealisationError as exc:
                failure = exc.failure

            part = partitions[failure.cell_id]
            if part.density <= 1:
                log(
                    f"  cell {failure.cell_id} already at density 1 and "
                    f"still failing (reason={failure.reason}, "
                    f"t={failure.timestep}, "
                    f"transits={failure.transit_robots}). Bailing."
                )
                return PathCollection()
            new_density = max(1, part.density // 2)
            log(
                f"  density retry: cell {failure.cell_id} "
                f"{part.density} -> {new_density} "
                f"(reason={failure.reason}, t={failure.timestep})"
            )
            part.update_density(new_density)
            for u, v, data in hlg.graph.edges(data=True):
                if data.get("cell_id") == failure.cell_id:
                    data["capacity"] = new_density

        log(f"  Exhausted {max_retries} density retries. Bailing.")
        return PathCollection()

    # ------------------------------------------------------------------
    # Path realisation
    # ------------------------------------------------------------------

    def _realise_paths(
        self,
        routing_sol: RoutingSolution,
        hlg: HighLevelGraph,
        partitions: List[Partition],
        robots,
        robot_radius: float,
        log,
    ) -> PathCollection:
        num_robots = len(robots)
        T = len(routing_sol[0])

        # Current geometric position per robot. At a cell boundary this
        # differs from the next timestep's computed entry; the gap is
        # materialised as two consecutive waypoints, and discopygal's
        # linear interpolation crosses the edge between them.
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
            timestep_segments = self._realise_timestep(
                t, routing_sol, hlg, partitions, current_pos, robot_radius,
                seed_base, log,
            )

            # Pad every robot's segment to the same length so the flattened
            # waypoint lists stay in lock-step.
            max_seg = max(
                (len(s) for s in timestep_segments.values()), default=1,
            )
            for r in range(num_robots):
                seg = timestep_segments.get(r, [current_pos[r]])
                while len(seg) < max_seg:
                    seg.append(seg[-1])
                robot_segments[r].append(seg)
                current_pos[r] = seg[-1]

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
        routing_sol: RoutingSolution,
        hlg: HighLevelGraph,
        partitions: List[Partition],
        current_pos: Dict[int, Tuple[float, float]],
        robot_radius: float,
        seed_base: int,
        log,
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Plan one router timestep. Raises RealisationError on PRM failure
        so the caller can retry with a lower cell density."""
        num_robots = len(current_pos)

        # Classify robots: holding (src == dst) or transiting via a cell.
        cell_transitions: Dict[int, List[Tuple[int, str, str]]] = defaultdict(
            list,
        )
        holding: List[int] = []

        for r in range(num_robots):
            src = routing_sol[r][t]
            dst = routing_sol[r][t + 1]
            if src == dst:
                holding.append(r)
                continue
            cell_id = hlg.graph.edges[src, dst].get("cell_id")
            cell_transitions[cell_id].append((r, src, dst))

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
                    transit_robots = [rr for rr, _, _ in transitions]
                    log(
                        f"  ad-hoc PRM: missing node position at t={t}, "
                        f"cell={cell_id}"
                    )
                    raise RealisationError(RealisationFailure(
                        cell_id=cell_id,
                        timestep=t,
                        transit_robots=transit_robots,
                        holders=list(holders_here),
                        reason="missing_node_position",
                    ))
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
                raise RealisationError(RealisationFailure(
                    cell_id=cell_id,
                    timestep=t,
                    transit_robots=transit_robots,
                    holders=list(holders_here),
                    reason=reason or "unknown",
                ))

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
        # Scale samples by sqrt(complexity), capped at 4x: elongated /
        # arc-carved cells get more samples without blowing up runtime.
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
        """Ad-hoc PRM with sample escalation (1x/2x/4x/8x). Invalid
        endpoints short-circuit; the caller handles them with a density retry."""
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

        # Cache key rounds pinned to 6dp so sub-precision drift between
        # timesteps (holders re-pinned from their last segment's final
        # waypoint) still hits the cache.
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

