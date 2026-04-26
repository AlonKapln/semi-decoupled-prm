import hashlib
import math
import os
import random
from collections import defaultdict
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

RND_SEED = 42


class RealisationError(Exception):
    def __init__(self, cell_id, timestep, reason, transit_robots, holders):
        super().__init__(
            f"cell={cell_id} t={timestep} reason={reason} "
            f"transits={transit_robots} holders={holders}"
        )
        self.cell_id = cell_id
        self.timestep = timestep
        self.reason = reason
        self.transit_robots = transit_robots
        self.holders = holders


class pPRMSolver(Solver):
    """
    Partitioned-PRM multi-robot solver.

    num_samples       PRM samples per cell.
    k_nearest         PRM neighbour connections per sample.
    time_horizon      Router time horizon; 0 or None auto-computes.
    max_cell_density  Upper bound on per-cell density used by grid refinement.
                      Lower values split large regions into more cells.
    """
    def __init__(
            self,
            num_samples: int = 50,
            k_nearest: int = 8,
            time_horizon: Optional[int] = None,
            max_cell_density: int = 100,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.time_horizon = time_horizon
        self.max_cell_density = max_cell_density

        self._arrangement = None
        self._hlg = None
        self._collision_checker = None
        # Reseeded from RND_SEED at the start of every solve.
        self._rng = random.Random(RND_SEED)

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
            "max_cell_density": ("Max cell density (grid refinement):", 100, int),
            **super().get_arguments(),
        }

    def _log(self, msg: str) -> None:
        if not getattr(self, "verbose", False):
            return
        writer = getattr(self, "writer", None)
        if writer:
            print(msg, file=writer)
        else:
            print(msg)

    @staticmethod
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

    @staticmethod
    def _point_in_partition(partition: Partition, x: float, y: float) -> bool:
        side = partition.polygon.bounded_side(Point_2(FT(x), FT(y)))
        return side != Bounded_side.ON_UNBOUNDED_SIDE

    @staticmethod
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

    @staticmethod
    def _total_ports(hlg: HighLevelGraph) -> int:
        """Each port appears in two cells' boundary maps; total is half the sum."""
        return sum(len(p) for p in hlg.cell_boundary_ports.values()) // 2

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _solve(self) -> PathCollection:
        scene = self.scene
        robots = scene.robots
        num_robots = len(robots)
        robot_radius = robots[0].radius.to_double()  # homogeneous radii
        self._rng = random.Random(RND_SEED)

        # Step 1: Minkowski-inflated free space + grid decomposition.
        self._log("Building free space and grid decomposition...")
        partitions, arrangement = partition_free_space_grid(
            scene, robot_radius, max_cell_density=self.max_cell_density,
        )
        self._arrangement = arrangement
        self._log(f"  {len(partitions)} cells")

        # Arc-exact obstacle checker. The polygonal cell approximation
        # chord-approximates inflated-obstacle arcs, so for curved faces
        # the checker overrides the polygon containment test.
        self._collision_checker = (
            ObjectCollisionDetection(scene.obstacles, robots[0])
            if scene.obstacles else None
        )

        # Step 2: high-level graph.
        self._log("Building high-level graph...")
        hlg = self._build_hlg(robots, partitions, robot_radius)
        self._hlg = hlg
        num_ports = self._total_ports(hlg)
        self._log(
            f"  V={hlg.graph.number_of_nodes()} "
            f"E={hlg.graph.number_of_edges()} ports={num_ports}"
        )

        for r in range(num_robots):
            if r not in hlg.start_cells or r not in hlg.goal_cells:
                self._log(f"  Robot {r} start/goal outside free space!")
                return PathCollection()

        # Step 3: cap densities by topology + sync to HLG edge capacities.
        # The router reads capacity from edge attributes, so the cap must
        # be propagated to every HLG edge, not just the partitions list.
        self._cap_densities_by_topology(hlg, partitions)
        self._sync_edge_capacities(hlg, partitions)

        self._save_snapshot(scene, partitions, hlg, robot_radius, num_ports)

        # Step 4: routing + realisation, with density-retry on failure.
        return self._route_with_retries(hlg, partitions, robots, robot_radius)

    def _build_hlg(
            self,
            robots,
            partitions: List[Partition],
            robot_radius: float,
    ) -> HighLevelGraph:
        robot_starts = [
            (r.start.x().to_double(), r.start.y().to_double()) for r in robots
        ]
        robot_goals = [
            (r.end.x().to_double(), r.end.y().to_double()) for r in robots
        ]
        return build_high_level_graph(
            partitions, robot_starts, robot_goals, robot_radius,
            checker=self._collision_checker,
        )

    def _cap_densities_by_topology(
            self,
            hlg: HighLevelGraph,
            partitions: List[Partition],
    ) -> None:
        """Cap each cell's density by its incident-node count.

        Every robot occupies exactly one HLG node at any timestep, so a
        cell's simultaneous-occupancy upper bound is the number of nodes
        incident to it (ports on its shared edges plus any starts/goals
        inside it). Disc-packing density may be much higher than that for
        large open cells, but the router can't exploit area without
        nodes to reserve.
        """
        for ci, partition in enumerate(partitions):
            node_cap = max(1, len(hlg.cell_incident_nodes.get(ci, [])))
            new_density = min(partition.density, node_cap)
            if new_density != partition.density:
                self._log(
                    f"  cell {ci}: density {partition.density} -> {new_density} "
                    f"(incident nodes={node_cap})"
                )
                partition.update_density(new_density)

    @staticmethod
    def _sync_edge_capacities(
            hlg: HighLevelGraph,
            partitions: List[Partition],
            cell_id: Optional[int] = None,
    ) -> None:
        """Push partition.density into HLG edge attributes. If cell_id is
        given, only that cell's edges; otherwise every edge."""
        for _u, _v, data in hlg.graph.edges(data=True):
            ci = data.get("cell_id")
            if ci is None:
                continue
            if cell_id is not None and ci != cell_id:
                continue
            data["capacity"] = partitions[ci].density

    def _save_snapshot(
            self,
            scene,
            partitions: List[Partition],
            hlg: HighLevelGraph,
            robot_radius: float,
            num_ports: int,
    ) -> None:
        """Best-effort cell-decomposition PNG dump."""
        try:
            stem = self._infer_scene_stem(scene)
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "visualizations",
                f"{stem}_d{self.max_cell_density}.png",
            )
            title = (
                f"{stem}, density={self.max_cell_density}, "
                f"{len(partitions)} cells, {num_ports} ports, "
                f"{len(scene.robots)} robots"
            )
            out = draw_cells(
                scene, partitions, hlg, robot_radius,
                save_path=save_path, title=title,
            )
            if out is not None:
                self._log(f"  cell snapshot -> {out}")
        except Exception as exc:
            self._log(f"  cell snapshot skipped: {exc}")

    def _route_with_retries(
            self,
            hlg: HighLevelGraph,
            partitions: List[Partition],
            robots,
            robot_radius: float,
    ) -> PathCollection:
        """Run the router and realisation; on realisation failure halve the
        offending cell's density (converges in ceil(log2(d)) attempts) and
        re-run. Cap at MAX_DENSITY_RETRIES extra attempts."""
        MAX_DENSITY_RETRIES = 3
        T = self.time_horizon or estimate_time_horizon(hlg)
        num_robots = len(robots)
        verbose = getattr(self, "verbose", False)

        for attempt in range(MAX_DENSITY_RETRIES + 1):
            self._log(f"Routing (T={T}, attempt={attempt})...")
            routing_sol = solve_prioritized(
                hlg, num_robots, T, verbose=verbose,
            )
            if routing_sol is None:
                self._log("  Prioritized router found no solution")
                return PathCollection()

            self._log("Realising geometric paths...")
            try:
                return self._realise_paths(
                    routing_sol, hlg, partitions, robots, robot_radius,
                )
            except RealisationError as exc:
                part = partitions[exc.cell_id]
                if part.density <= 1:
                    self._log(
                        f"  cell {exc.cell_id} already at density 1 and "
                        f"still failing (reason={exc.reason}, "
                        f"t={exc.timestep}, "
                        f"transits={exc.transit_robots}). Bailing."
                    )
                    return PathCollection()
                new_density = max(1, part.density // 2)
                self._log(
                    f"  density retry: cell {exc.cell_id} "
                    f"{part.density} -> {new_density} "
                    f"(reason={exc.reason}, t={exc.timestep})"
                )
                part.update_density(new_density)
                self._sync_edge_capacities(hlg, partitions, cell_id=exc.cell_id)

        self._log(f"  Exhausted {MAX_DENSITY_RETRIES} density retries. Bailing.")
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

        for t in range(T - 1):
            timestep_segments = self._realise_timestep(
                t, routing_sol, hlg, partitions, current_pos, robot_radius,
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

        self._log(f"  {num_robots} paths, {max_len} waypoints each")
        return pc

    def _realise_timestep(
            self,
            t: int,
            routing_sol: RoutingSolution,
            hlg: HighLevelGraph,
            partitions: List[Partition],
            current_pos: Dict[int, Tuple[float, float]],
            robot_radius: float,
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
                if self._point_in_partition(p, px, py):
                    cell_holding[ci].append(r)
                    break

        # Holders' segments are trivial single-point lists.
        timestep_segments: Dict[int, List[Tuple[float, float]]] = {
            r: [current_pos[r]] for r in holding
        }

        # For each cell with active transits, build the joint roadmap and
        # extract per-robot waypoint lists from the resulting joint path.
        for cell_id, transitions in cell_transitions.items():
            self._plan_cell_timestep(
                t, cell_id, transitions,
                cell_holding.get(cell_id, []),
                hlg, partitions[cell_id], current_pos, robot_radius,
                timestep_segments,
            )

        return timestep_segments

    def _plan_cell_timestep(
            self,
            t: int,
            cell_id: int,
            transitions: List[Tuple[int, str, str]],
            holders_here: List[int],
            hlg: HighLevelGraph,
            partition: Partition,
            current_pos: Dict[int, Tuple[float, float]],
            robot_radius: float,
            timestep_segments: Dict[int, List[Tuple[float, float]]],
    ) -> None:
        """Plan one cell at one timestep; mutate timestep_segments in place."""
        transit_robots = [r for r, _, _ in transitions]

        transit_entries: List[Tuple[float, float]] = []
        transit_exits: List[Tuple[float, float]] = []
        for _r, src, dst in transitions:
            entry = self._node_position_in_cell(hlg, cell_id, src)
            ex = self._node_position_in_cell(hlg, cell_id, dst)
            if entry is None or ex is None:
                self._log(
                    f"  ad-hoc PRM: missing node position at t={t}, "
                    f"cell={cell_id}"
                )
                raise RealisationError(
                    cell_id=cell_id, timestep=t,
                    reason="missing_node_position",
                    transit_robots=transit_robots,
                    holders=list(holders_here),
                )
            transit_entries.append(entry)
            transit_exits.append(ex)

        pinned = [current_pos[r] for r in holders_here]
        base_samples = self._effective_num_samples(partition)
        cfg_path, reason = self._plan_adhoc(
            partition, transit_entries, transit_exits, pinned,
            robot_radius, base_samples=base_samples,
        )
        if cfg_path is None:
            self._log(
                f"  ad-hoc PRM failed at t={t}, cell={cell_id}, "
                f"transits={transit_robots}, holders={holders_here}, "
                f"reason={reason}"
            )
            raise RealisationError(
                cell_id=cell_id, timestep=t,
                reason=reason or "unknown",
                transit_robots=transit_robots,
                holders=list(holders_here),
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

    def _effective_num_samples(self, partition: Partition) -> int:
        """
        We increase the number of samples for the ad hoc PRM in a Partition if the Partition is condered complex.
        :param partition: partition for which we want to build an ad hoc PRM map.
        :return: num samples for the ad hoc PRM
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
            base_samples: Optional[int] = None,
    ) -> Tuple[Optional[List[Tuple[Tuple[float, float], ...]]], Optional[str]]:
        """Ad-hoc PRM with sample escalation (1x/2x/4x/8x). Invalid
        endpoints short-circuit; the caller handles them with a density retry.

        Each call draws its own child seed from the solver's master RNG,
        so successive ad-hoc roadmaps get independent random streams while
        the whole solve stays reproducible from RND_SEED.
        """
        rng = random.Random(self._rng.randrange(1 << 32))
        base = base_samples if base_samples is not None else self.num_samples
        last_reason: Optional[str] = None
        for nsamples in (base, base * 2, base * 4, base * 8):
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
            last_reason = result.reason
            if result.reason in ("entry_infeasible", "exit_infeasible"):
                return None, result.reason
            if result.graph is None:
                continue
            entry_cfg, exit_cfg = result.entry_cfg, result.exit_cfg
            if entry_cfg == exit_cfg:
                return [entry_cfg], None
            try:
                path = nx.shortest_path(
                    result.graph, entry_cfg, exit_cfg, weight="weight",
                )
                return path, None
            except nx.NetworkXNoPath:
                last_reason = "no_path"
                continue
        return None, last_reason
