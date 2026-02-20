write# Multi-Robot Grid-Based Flow Solver - Documentation

## Overview

This solver implements a multi-robot motion planning algorithm using grid-based space decomposition and multi-commodity flow optimization. Unlike the existing solvers in this project (which handle single robots), this solver coordinates multiple robots simultaneously while respecting density constraints.

## Architecture

### 1. Grid Decomposition

The workspace is divided into a regular grid where each cell is `cell_size × cell_size` units (default: 5×5).

**Key Properties:**
- **4-connectivity**: Each cell connects only to its vertical and horizontal neighbors (not diagonal)
- **GridCell class**: Stores cell boundaries, neighbors, and center point
- **Spatial indexing**: Fast lookup to find which cell contains a given point

**Example Grid Layout (4×4):**
```
+----+----+----+----+
| 12 | 13 | 14 | 15 |
+----+----+----+----+
|  8 |  9 | 10 | 11 |
+----+----+----+----+
|  4 |  5 |  6 |  7 |
+----+----+----+----+
|  0 |  1 |  2 |  3 |
+----+----+----+----+
```

Each cell has at most 4 neighbors (edge cells have fewer).

### 2. Multi-Commodity Flow Formulation

The solver formulates the multi-robot planning problem as a **time-expanded multi-commodity flow problem**.

#### State Space
- **Nodes**: (cell_id, timestep, robot_id) tuples
- **Time horizon**: T timesteps (estimated or provided)
- **Commodities**: Each robot is a separate commodity that must flow from its start to goal

#### Decision Variables

```
flow[r][c][t] ∈ {0, 1}
```
- `r`: robot index (0 to num_robots-1)
- `c`: cell index (0 to num_cells-1)  
- `t`: time index (0 to T)
- `flow[r][c][t] = 1` if robot r is in cell c at time t

#### Constraints

**1. Single Location Constraint**  
Each robot must be in exactly one cell at each time:
```
∀r, t:  Σ_c flow[r][c][t] = 1
```

**2. Initial Condition**  
Robot r starts in its designated start cell:
```
∀r:  flow[r][start_cell[r]][0] = 1
```

**3. Goal Condition**  
Robot r must reach its goal by time T:
```
∀r:  flow[r][goal_cell[r]][T] = 1
```

**4. Flow Conservation (Movement)**  
If robot is in cell c at time t, it can stay or move to an adjacent cell at t+1:
```
∀r, c, t < T:  flow[r][c][t] ≤ Σ_{c' ∈ neighbors(c) ∪ {c}} flow[r][c'][t+1]
```

**5. Density Constraint**  
At most `max_density` robots can occupy a cell at any time:
```
∀c, t:  Σ_r flow[r][c][t] ≤ max_density
```

#### Objective Function

Minimize total time (or path length):
```
minimize:  Σ_r Σ_c Σ_t flow[r][c][t]
```

This encourages robots to reach goals quickly and take direct paths.

### 3. Flow Control Strategies

The solver implements three different flow control strategies:

#### Strategy 1: ILP (Optimal)

**Method**: Solve the full multi-commodity flow ILP using OR-Tools

**Pros:**
- Globally optimal solution
- Handles complex coordination scenarios
- Respects all density constraints exactly

**Cons:**
- Computationally expensive (exponential in worst case)
- Requires OR-Tools library
- May not scale to many robots or long time horizons

**Use when**: You need optimal solutions and have <10 robots with moderate workspace

**Implementation**: Uses Google OR-Tools SCIP solver with binary integer programming

#### Strategy 2: Sequential Planning

**Method**: Plan robots one at a time, treating previously planned paths as dynamic obstacles

**Algorithm:**
1. Order robots arbitrarily (0, 1, 2, ...)
2. For each robot i:
   - Find shortest path avoiding cells reserved by robots 0...i-1
   - Reserve cells used by this path for future robots
3. Return combined paths

**Pros:**
- Fast and scalable
- Always finds a solution if one exists (without conflicts)
- Simple to implement and understand

**Cons:**
- Suboptimal (order matters!)
- Later robots may have very long paths
- Can fail even when a solution exists with different ordering

**Use when**: You have many robots or need fast solutions

**Implementation**: A* search on time-expanded graph with reservation table

#### Strategy 3: Priority-Based Planning

**Method**: Sequential planning with intelligent ordering

**Algorithm:**
1. Compute priority for each robot (e.g., by path distance)
2. Sort robots by priority (longer paths first)
3. Apply sequential planning in priority order

**Pros:**
- Better than arbitrary sequential ordering
- Fast like sequential
- More robust to ordering issues

**Cons:**
- Still suboptimal
- Priority heuristic may not always help
- Requires good priority function

**Use when**: Sequential fails due to ordering, but ILP is too slow

**Implementation**: Distance-based priority + sequential planning

### 4. Time Horizon Estimation

The solver auto-computes time horizon if not provided:

```python
max_distance = max over all robots (
    manhattan_distance(start_cell, goal_cell)
)
time_horizon = max(10, max_distance × 2)
```

The 2× factor provides buffer for conflicts and detours.

## Usage Examples

### Basic Usage

```python
from discopygal.solvers_infra import Scene, RobotDisc
from discopygal.bindings import Point_2, FT
from multi_robot_flow_solver import MultiRobotFlowSolver

# Create scene with robots
scene = Scene()
robot1 = RobotDisc(
    start=Point_2(FT(1), FT(1)),
    end=Point_2(FT(18), FT(18)),
    radius=0.3
)
scene.add_robot(robot1)
# ... add more robots and obstacles ...

# Create solver
solver = MultiRobotFlowSolver(
    cell_size=5.0,
    max_density=1,
    time_horizon=20,
    flow_control_strategy='sequential',
    verbose=True
)

# Solve
solver.load_scene(scene)
paths = solver.solve()

# Access results
for robot, path in paths.paths.items():
    print(f"Path for robot: {len(path.points)} waypoints")
```

### Advanced Configuration

```python
# High-density scenario (allow 2 robots per cell)
solver = MultiRobotFlowSolver(
    cell_size=3.0,        # Finer grid
    max_density=2,        # Allow more overlap
    time_horizon=50,      # Longer planning horizon
    flow_control_strategy='priority',
    verbose=True
)
```

### Testing Different Strategies

```python
strategies = ['ilp', 'sequential', 'priority']

for strategy in strategies:
    solver = MultiRobotFlowSolver(
        cell_size=5.0,
        max_density=1,
        flow_control_strategy=strategy
    )
    solver.load_scene(scene)
    paths = solver.solve()
    print(f"{strategy}: {len(paths.paths)} robots planned")
```

## Algorithm Complexity

### Space Complexity
- Grid cells: O(W×H / cell_size²) where W,H are workspace dimensions
- ILP variables: O(R × C × T) where R=robots, C=cells, T=time horizon
- Sequential: O(C × T) for reservation table

### Time Complexity
- Grid construction: O(C) where C = number of cells
- ILP solving: O(2^(R×C×T)) worst case, but practical with <1000 variables
- Sequential: O(R × C × T × log(C×T)) for R robots with A* search
- Priority: Same as sequential + O(R log R) for sorting

## Flow Control Mechanism Details

### Cell Reservation System (Sequential/Priority)

The reservation table tracks which robot occupies which cell at each time:

```python
reserved_cells = {
    (cell_id, time): robot_id
}
```

When planning for robot i:
1. Run A* from start to goal on (cell, time) state space
2. Check reservation table before expanding neighbors
3. Skip cells with `reserved_cells[(cell, time)] != None`
4. After finding path, add reservations for robot i

### Density Enforcement

**In ILP:** Hard constraint in optimization problem
```
Σ_r flow[r][c][t] ≤ max_density
```

**In Sequential/Priority:** Check during A* expansion
```python
count = sum(1 for (c, t), _ in reserved_cells.items() 
           if c == next_cell_id and t == next_time)
if count >= max_density:
    continue  # Skip this neighbor
```

## Suggestions for Flow Control Improvements

### 1. Velocity Obstacles Integration

Combine grid-based planning with velocity obstacles:
- Use grid for high-level planning (which cells to visit)
- Use velocity obstacles for low-level control (exact trajectories)
- Enables smoother, more realistic motion

### 2. Reactive Flow Fields

Create flow fields that guide robots dynamically:
- Compute potential field for each robot toward its goal
- At each timestep, robots move along their gradient
- Adjust fields based on current robot positions (reactive)
- Handles dynamic replanning naturally

### 3. Token-Based Flow Control

Introduce tokens for cell access:
- Each cell has `max_density` tokens
- Robot must acquire token to enter cell
- Releases token when leaving
- Prevents deadlocks with proper token circulation

### 4. Hierarchical Flow Planning

Multi-resolution approach:
- Coarse level: Plan with large cells and long time steps
- Fine level: Refine within each coarse cell
- Reduces problem size while maintaining detail where needed

### 5. Time Windows

Instead of discrete timesteps, use time windows:
- Allow continuous time within [t, t+dt]
- Robots reserve time intervals, not just timesteps
- More flexible scheduling

### 6. Priority Inheritance

Dynamic priority adjustment:
- Initially assign priorities based on heuristic
- If robot i blocks robot j, temporarily boost j's priority
- Prevents priority inversion and starvation

### 7. Coupled Planning with Breakpoints

Hybrid between sequential and coupled:
- Identify critical "bottleneck" cells (e.g., corridors)
- Plan bottleneck cells with full ILP
- Plan open areas with sequential
- Balance optimality and scalability

## Installation Requirements

```bash
# Core requirements (already in your project)
pip install discopygal_taucgl-1.4.5-py3-none-any.whl

# For ILP strategy (optional but recommended)
pip install ortools

# For visualization (optional)
pip install matplotlib
```

## Testing

Run the test suite:
```bash
python test_multi_robot_flow.py
```

This will test:
- ILP strategy (if OR-Tools available)
- Sequential strategy
- Priority-based strategy
- Corridor scenario (coordination test)
- Strategy comparison

## Integration with Existing Code

The solver follows the discopygal framework:

1. **Inherits from `Solver`** - Compatible with existing infrastructure
2. **Uses standard types** - `Point_2`, `PathCollection`, `Scene`, etc.
3. **Implements required methods** - `_solve()`, `get_arguments()`
4. **Respects conventions** - `verbose`, `writer`, `_bounding_box`

Can be used interchangeably with other solvers in the project.

## Limitations and Future Work

### Current Limitations
- Assumes point robots (discs) - doesn't account for robot shape in grid
- Fixed grid resolution (no adaptive refinement)
- Time discretization may miss continuous-time conflicts
- No dynamic obstacles
- No robot dynamics (assumes instant velocity changes)

### Future Extensions
1. **Adaptive grid**: Refine cells near obstacles or dense regions
2. **Robot shape**: Include rotation for non-circular robots
3. **Dynamic obstacles**: Extend time-expanded graph to include moving obstacles
4. **Kinodynamic constraints**: Add velocity/acceleration limits
5. **Distributed solving**: Parallelize for many robots
6. **Learning**: Use ML to predict good priority orderings or time horizons
7. **Anytime behavior**: Return best solution found so far if interrupted

## References

### Multi-Commodity Flow
- Ahuja, R.K., Magnanti, T.L., & Orlin, J.B. (1993). Network Flows. Prentice Hall.
- Even, S., Itai, A., & Shamir, A. (1976). On the complexity of time table and multi-commodity flow problems. SIAM Journal on Computing.

### Multi-Robot Motion Planning  
- Čáp, M., et al. (2015). Prioritized Planning Algorithms for Trajectory Coordination of Multiple Mobile Robots. IEEE Transactions on Automation Science and Engineering.
- Sharon, G., et al. (2015). Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence.
- Yu, J., & LaValle, S.M. (2013). Structure and intractability of optimal multi-robot path planning on graphs. AAAI.

### Velocity Obstacles & Flow Fields
- Van den Berg, J., et al. (2011). Reciprocal n-body collision avoidance. Robotics Research.
- Trautman, P., & Krause, A. (2010). Unfreezing the robot: Navigation in dense, interacting crowds. IROS.

## Support and Contact

For questions or issues with this solver, please refer to:
- Test file: `test_multi_robot_flow.py` - Contains usage examples
- Main implementation: `multi_robot_flow_solver.py` - Well-commented code
- Discopygal docs: Check the framework documentation for base classes

---

**Version**: 1.0  
**Created**: February 2026  
**Author**: Multi-Robot Planning Project

