"""
Multi-Robot Grid-Based Flow Solver

This solver decomposes the configuration space into a grid (default 5x5 units per cell),
connects cells with 4-connectivity (vertical/horizontal only), and solves a multi-commodity
flow problem with density constraints to coordinate multiple robots.

Algorithm:
1. Discretize workspace into grid cells
2. Build time-expanded graph with (cell, timestep, robot) nodes
3. Formulate multi-commodity flow problem with capacity constraints
4. Solve using Integer Linear Programming (ILP)
5. Extract paths for each robot from flow solution
"""

import math
import networkx as nx
from typing import List, Dict, Tuple, Set, Union
from dataclasses import dataclass, field

from discopygal.bindings import Point_2, FT, Segment_2
from discopygal.geometry_utils import collision_detection, bounding_boxes
from discopygal.solvers_infra import Solver, PathPoint, Path, PathCollection

# Optional: We'll implement both with and without OR-Tools
try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("Warning: OR-Tools not available. Install with: pip install ortools")


@dataclass
class GridCell:
    """Represents a single cell in the grid decomposition."""
    cell_id: int
    row: int
    col: int
    min_x: Union[float, FT]
    max_x: Union[float, FT]
    min_y: Union[float, FT]
    max_y: Union[float, FT]
    neighbors: List[int] = field(default_factory=list)  # Adjacent cell IDs (4-connectivity)

    @property
    def center(self) -> Point_2:
        """Return the center point of the cell."""
        center_x = (self.min_x + self.max_x) / 2.0
        center_y = (self.min_y + self.max_y) / 2.0
        return Point_2(FT(center_x), FT(center_y))

    def contains_point(self, point: Point_2) -> bool:
        """Check if a point is inside this cell."""
        x = point.x().to_double() if hasattr(point.x(), 'to_double') else float(point.x())
        y = point.y().to_double() if hasattr(point.y(), 'to_double') else float(point.y())
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)


class MultiRobotFlowSolver(Solver):
    """
    Multi-robot motion planning solver using grid-based decomposition and multi-commodity flow.

    :param cell_size: Size of each grid cell (default: 5.0 units)
    :param max_density: Maximum number of robots allowed in a cell at each timestep (default: 1)
    :param time_horizon: Maximum number of timesteps to consider (default: auto-computed)
    :param flow_control_strategy: Strategy for flow control ('ilp', 'sequential', 'priority')
    """

    def __init__(self, cell_size=5.0, max_density=1, time_horizon=None,
                 flow_control_strategy='ilp', **kwargs):
        super().__init__(**kwargs)
        self.cell_size = FT(cell_size)
        self.max_density = max_density
        self.time_horizon = time_horizon
        self.flow_control_strategy = flow_control_strategy

        # Internal data structures
        self.grid_cells: Dict[int, GridCell] = {}
        self.num_rows = 0
        self.num_cols = 0
        self.time_expanded_graph = None
        self.robot_start_cells: Dict[int, int] = {}  # robot_idx -> cell_id
        self.robot_goal_cells: Dict[int, int] = {}   # robot_idx -> cell_id

    @classmethod
    def get_arguments(cls):
        """
        Return a list of arguments and their description, defaults and types.
        Can be used by a GUI to generate fields dynamically.

        :return: arguments dict
        :rtype: :class:`dict`
        """
        args = {
            'cell_size': ('Size of each grid cell:', 5.0, float),
            'max_density': ('Maximum robots per cell per timestep:', 1, int),
            'time_horizon': ('Maximum timesteps (None for auto):', None, int),
            'flow_control_strategy': ('Flow control strategy (ilp/sequential/priority):', 'ilp', str),
        }
        args.update(super().get_arguments())
        return args

    def _solve(self):
        """
        Main solving method: decompose space, build flow problem, solve, extract paths.

        :return: path collection for all robots
        :rtype: :class:`~discopygal.solvers_infra.PathCollection`
        """
        # Step 1: Initialize grid decomposition
        if self.verbose:
            print(f"Initializing grid with cell_size={self.cell_size}...", file=self.writer)
        self._initialize_grid()

        # Step 2: Map robot start/goal positions to cells
        if not self._map_robots_to_cells():
            if self.verbose:
                print("Failed to map robots to valid cells", file=self.writer)
            return PathCollection()

        # Step 3: Estimate time horizon if not provided
        if self.time_horizon is None:
            self._estimate_time_horizon()

        if self.verbose:
            print(f"Grid: {self.num_rows}x{self.num_cols} = {len(self.grid_cells)} cells",
                  file=self.writer)
            print(f"Robots: {len(self.scene.robots)}, Time horizon: {self.time_horizon}",
                  file=self.writer)

        # Step 4: Solve based on strategy
        if self.flow_control_strategy == 'ilp':
            return self._solve_with_ilp()
        elif self.flow_control_strategy == 'sequential':
            return self._solve_sequential()
        elif self.flow_control_strategy == 'priority':
            return self._solve_priority_based()
        else:
            raise ValueError(f"Unknown flow control strategy: {self.flow_control_strategy}")

    def _initialize_grid(self):
        """Create grid cells based on workspace bounding box."""
        # Get or compute bounding box
        if self._bounding_box is None:
            self._bounding_box = bounding_boxes.calc_scene_bounding_box(self.scene)

        min_x, max_x, min_y, max_y = self._bounding_box

        # Calculate number of cells in each dimension
        width = max_x - min_x
        height = max_y - min_y
        self.num_cols = max(1, int(math.ceil(width / self.cell_size)))
        self.num_rows = max(1, int(math.ceil(height / self.cell_size)))

        # Create cells
        cell_id = 0
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_min_x = min_x + col * self.cell_size
                cell_max_x = min(max_x, min_x + (col + 1) * self.cell_size)
                cell_min_y = min_y + row * self.cell_size
                cell_max_y = min(max_y, min_y + (row + 1) * self.cell_size)

                cell = GridCell(
                    cell_id=cell_id,
                    row=row,
                    col=col,
                    min_x=cell_min_x,
                    max_x=cell_max_x,
                    min_y=cell_min_y,
                    max_y=cell_max_y
                )
                self.grid_cells[cell_id] = cell
                cell_id += 1

        # Build adjacency (4-connectivity: up, down, left, right)
        for cell_id, cell in self.grid_cells.items():
            # Right neighbor
            if cell.col < self.num_cols - 1:
                right_id = cell_id + 1
                cell.neighbors.append(right_id)
            # Left neighbor
            if cell.col > 0:
                left_id = cell_id - 1
                cell.neighbors.append(left_id)
            # Up neighbor (assuming row 0 is bottom)
            if cell.row < self.num_rows - 1:
                up_id = cell_id + self.num_cols
                cell.neighbors.append(up_id)
            # Down neighbor
            if cell.row > 0:
                down_id = cell_id - self.num_cols
                cell.neighbors.append(down_id)

    def _find_cell_for_point(self, point: Point_2) -> int:
        """Find which cell contains the given point. Returns -1 if not found."""
        for cell_id, cell in self.grid_cells.items():
            if cell.contains_point(point):
                return cell_id
        return -1

    def _map_robots_to_cells(self) -> bool:
        """Map each robot's start and goal to grid cells."""
        for robot_idx, robot in enumerate(self.scene.robots):
            start_cell = self._find_cell_for_point(robot.start)
            goal_cell = self._find_cell_for_point(robot.end)

            if start_cell == -1:
                if self.verbose:
                    print(f"Robot {robot_idx} start position not in grid", file=self.writer)
                return False
            if goal_cell == -1:
                if self.verbose:
                    print(f"Robot {robot_idx} goal position not in grid", file=self.writer)
                return False

            self.robot_start_cells[robot_idx] = start_cell
            self.robot_goal_cells[robot_idx] = goal_cell

        return True

    def _estimate_time_horizon(self):
        """Estimate time horizon based on maximum individual path length."""
        max_distance = 0
        for robot_idx in range(len(self.scene.robots)):
            start_cell = self.grid_cells[self.robot_start_cells[robot_idx]]
            goal_cell = self.grid_cells[self.robot_goal_cells[robot_idx]]

            # Manhattan distance in grid cells
            distance = abs(start_cell.row - goal_cell.row) + abs(start_cell.col - goal_cell.col)
            max_distance = max(max_distance, distance)

        # Add buffer for potential conflicts (2x the longest path)
        self.time_horizon = max(10, max_distance * 2)

    # =========================================================================
    # ILP-based Multi-Commodity Flow Solver
    # =========================================================================

    def _solve_with_ilp(self):
        """Solve using Integer Linear Programming (multi-commodity flow)."""
        if not HAS_ORTOOLS:
            if self.verbose:
                print("OR-Tools not available, falling back to sequential solver",
                      file=self.writer)
            return self._solve_sequential()

        if self.verbose:
            print("Building ILP formulation...", file=self.writer)

        # Create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            if self.verbose:
                print("SCIP solver not available", file=self.writer)
            return PathCollection()

        num_robots = len(self.scene.robots)
        num_cells = len(self.grid_cells)
        T = self.time_horizon

        # Decision variables: flow[robot][cell][time] ∈ {0, 1}
        # flow[r][c][t] = 1 if robot r is in cell c at time t
        flow = {}
        for r in range(num_robots):
            flow[r] = {}
            for c in range(num_cells):
                flow[r][c] = {}
                for t in range(T + 1):
                    flow[r][c][t] = solver.BoolVar(f'flow_r{r}_c{c}_t{t}')

        # Objective: minimize total time (sum of all flows)
        objective = solver.Objective()
        for r in range(num_robots):
            for c in range(num_cells):
                for t in range(T + 1):
                    objective.SetCoefficient(flow[r][c][t], 1)
        objective.SetMinimization()

        # Constraint 1: Each robot must be in exactly one cell at each time
        for r in range(num_robots):
            for t in range(T + 1):
                constraint = solver.Constraint(1, 1)
                for c in range(num_cells):
                    constraint.SetCoefficient(flow[r][c][t], 1)

        # Constraint 2: Initial conditions (robot r starts in its start cell)
        for r in range(num_robots):
            start_cell = self.robot_start_cells[r]
            solver.Add(flow[r][start_cell][0] == 1)

        # Constraint 3: Goal conditions (robot r must reach goal by time T)
        for r in range(num_robots):
            goal_cell = self.robot_goal_cells[r]
            solver.Add(flow[r][goal_cell][T] == 1)

        # Constraint 4: Flow conservation (movement constraints)
        # If robot is in cell c at time t, it can stay or move to a neighbor at t+1
        for r in range(num_robots):
            for t in range(T):
                for c in range(num_cells):
                    # flow[r][c][t] <= sum of (flow[r][c][t+1] + flow[r][neighbor][t+1])
                    cell = self.grid_cells[c]
                    reachable_cells = [c] + cell.neighbors  # Can stay or move to neighbor

                    # If in cell c at time t, must be in reachable cell at t+1
                    constraint = solver.Constraint(-solver.infinity(), 0)
                    constraint.SetCoefficient(flow[r][c][t], -1)
                    for next_c in reachable_cells:
                        constraint.SetCoefficient(flow[r][next_c][t + 1], 1)

        # Constraint 5: Density constraint (max_density robots per cell per time)
        for c in range(num_cells):
            for t in range(T + 1):
                constraint = solver.Constraint(0, self.max_density)
                for r in range(num_robots):
                    constraint.SetCoefficient(flow[r][c][t], 1)

        if self.verbose:
            print(f"ILP has {solver.NumVariables()} variables and {solver.NumConstraints()} constraints",
                  file=self.writer)
            print("Solving ILP...", file=self.writer)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            if self.verbose:
                print(f"Solution found! Status: {'OPTIMAL' if status == pywraplp.Solver.OPTIMAL else 'FEASIBLE'}",
                      file=self.writer)

            # Extract paths from flow solution
            return self._extract_paths_from_flow(flow, T)
        else:
            if self.verbose:
                print(f"No solution found. Status: {status}", file=self.writer)
            return PathCollection()

    def _extract_paths_from_flow(self, flow, T):
        """Extract robot paths from the flow solution."""
        path_collection = PathCollection()
        num_robots = len(self.scene.robots)

        for r in range(num_robots):
            robot = self.scene.robots[r]
            path_points = []

            # Add actual start position
            path_points.append(PathPoint(robot.start))

            # Find cells where robot is present at each timestep
            prev_cell_id = self.robot_start_cells[r]
            for t in range(1, T + 1):
                # Find which cell has flow = 1 for this robot at time t
                current_cell_id = None
                for c in range(len(self.grid_cells)):
                    if flow[r][c][t].solution_value() > 0.5:  # Binary variable = 1
                        current_cell_id = c
                        break

                # Add cell center if robot moved to a different cell
                if current_cell_id is not None and current_cell_id != prev_cell_id:
                    cell_center = self.grid_cells[current_cell_id].center
                    path_points.append(PathPoint(cell_center))
                    prev_cell_id = current_cell_id

            # Add actual goal position
            path_points.append(PathPoint(robot.end))

            # Create path and add to collection
            path = Path(path_points)
            path_collection.add_robot_path(robot, path)

        return path_collection

    # =========================================================================
    # Alternative Flow Control Strategies
    # =========================================================================

    def _solve_sequential(self):
        """
        Sequential planning: plan for each robot in sequence, treating previous
        robots' paths as dynamic obstacles.
        """
        if self.verbose:
            print("Using sequential planning strategy...", file=self.writer)

        path_collection = PathCollection()
        reserved_cells = {}  # (cell_id, time) -> robot_idx

        for robot_idx, robot in enumerate(self.scene.robots):
            if self.verbose:
                print(f"Planning for robot {robot_idx}...", file=self.writer)

            # Find shortest path for this robot avoiding reserved cells
            path = self._find_path_avoiding_reservations(
                robot_idx, reserved_cells
            )

            if path is None:
                if self.verbose:
                    print(f"No path found for robot {robot_idx}", file=self.writer)
                return PathCollection()

            path_collection.add_robot_path(robot, path)

            # Reserve cells used by this robot's path
            self._reserve_path_cells(robot_idx, path, reserved_cells)

        return path_collection

    def _find_path_avoiding_reservations(self, robot_idx, reserved_cells):
        """
        Find a path for a single robot using A* on the time-expanded grid,
        avoiding cells reserved by other robots.
        """
        import heapq

        start_cell = self.robot_start_cells[robot_idx]
        goal_cell = self.robot_goal_cells[robot_idx]

        # A* search on (cell_id, time) state space
        # State: (f_score, g_score, cell_id, time, path)
        start_state = (0, 0, start_cell, 0, [start_cell])
        pq = [start_state]
        visited = set()

        def heuristic(cell_id):
            """Manhattan distance to goal."""
            cell = self.grid_cells[cell_id]
            goal = self.grid_cells[goal_cell]
            return abs(cell.row - goal.row) + abs(cell.col - goal.col)

        while pq:
            f_score, g_score, cell_id, time, path = heapq.heappop(pq)

            # Check if reached goal
            if cell_id == goal_cell:
                # Convert cell path to PathPoint path
                path_points = [PathPoint(self.scene.robots[robot_idx].start)]
                for c_id in path[1:]:
                    path_points.append(PathPoint(self.grid_cells[c_id].center))
                path_points.append(PathPoint(self.scene.robots[robot_idx].end))
                return Path(path_points)

            # Skip if already visited this state
            state_key = (cell_id, time)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Check time limit
            if time >= self.time_horizon:
                continue

            # Expand neighbors (including staying in same cell)
            cell = self.grid_cells[cell_id]
            next_cells = [cell_id] + cell.neighbors

            for next_cell_id in next_cells:
                next_time = time + 1

                # Check if cell is reserved
                if (next_cell_id, next_time) in reserved_cells:
                    continue

                # Check density constraint (count robots already at this cell-time)
                count = sum(1 for (c, t), _ in reserved_cells.items()
                           if c == next_cell_id and t == next_time)
                if count >= self.max_density:
                    continue

                new_g = g_score + 1
                new_f = new_g + heuristic(next_cell_id)
                new_path = path + [next_cell_id]

                heapq.heappush(pq, (new_f, new_g, next_cell_id, next_time, new_path))

        return None  # No path found

    def _reserve_path_cells(self, robot_idx, path, reserved_cells):
        """Mark cells as reserved based on a robot's path."""
        # Extract cell sequence from path
        # First point is start, last is goal, middle points are cell centers
        for time, point in enumerate(path.points):
            cell_id = self._find_cell_for_point(point.location)
            if cell_id != -1:
                reserved_cells[(cell_id, time)] = robot_idx

    def _solve_priority_based(self):
        """
        Priority-based planning: similar to sequential, but with priority ordering.
        Higher priority robots are planned first.
        """
        if self.verbose:
            print("Using priority-based planning strategy...", file=self.writer)

        # Simple priority: longer paths get higher priority
        priorities = []
        for robot_idx, robot in enumerate(self.scene.robots):
            start_cell = self.grid_cells[self.robot_start_cells[robot_idx]]
            goal_cell = self.grid_cells[self.robot_goal_cells[robot_idx]]
            distance = abs(start_cell.row - goal_cell.row) + abs(start_cell.col - goal_cell.col)
            priorities.append((distance, robot_idx))

        # Sort by distance (descending)
        priorities.sort(reverse=True)

        if self.verbose:
            print(f"Robot planning order: {[r for _, r in priorities]}", file=self.writer)

        # Plan in priority order
        path_collection = PathCollection()
        reserved_cells = {}
        robot_to_path = {}

        for _, robot_idx in priorities:
            robot = self.scene.robots[robot_idx]
            path = self._find_path_avoiding_reservations(robot_idx, reserved_cells)

            if path is None:
                if self.verbose:
                    print(f"No path found for robot {robot_idx}", file=self.writer)
                return PathCollection()

            robot_to_path[robot_idx] = path
            self._reserve_path_cells(robot_idx, path, reserved_cells)

        # Add paths in original robot order
        for robot_idx, robot in enumerate(self.scene.robots):
            path_collection.add_robot_path(robot, robot_to_path[robot_idx])

        return path_collection


# =========================================================================
# Utility Functions for Visualization and Debugging
# =========================================================================

def visualize_grid(solver: MultiRobotFlowSolver):
    """
    Helper function to visualize the grid decomposition.
    Returns a list of segments representing grid lines.
    """
    segments = []
    for cell in solver.grid_cells.values():
        # Add cell boundaries as segments
        p1 = Point_2(FT(cell.min_x), FT(cell.min_y))
        p2 = Point_2(FT(cell.max_x), FT(cell.min_y))
        p3 = Point_2(FT(cell.max_x), FT(cell.max_y))
        p4 = Point_2(FT(cell.min_x), FT(cell.max_y))

        segments.extend([
            Segment_2(p1, p2),
            Segment_2(p2, p3),
            Segment_2(p3, p4),
            Segment_2(p4, p1)
        ])

    return segments


def print_grid_info(solver: MultiRobotFlowSolver):
    """Print debugging information about the grid."""
    print(f"Grid decomposition: {solver.num_rows} rows × {solver.num_cols} cols = {len(solver.grid_cells)} cells")
    print(f"Cell size: {solver.cell_size}")
    print(f"Bounding box: {solver._bounding_box}")
    print("\nRobot assignments:")
    for robot_idx in range(len(solver.scene.robots)):
        start_cell = solver.robot_start_cells[robot_idx]
        goal_cell = solver.robot_goal_cells[robot_idx]
        print(f"  Robot {robot_idx}: start_cell={start_cell}, goal_cell={goal_cell}")
    print(f"\nMax density per cell: {solver.max_density}")
    print(f"Time horizon: {solver.time_horizon}")

