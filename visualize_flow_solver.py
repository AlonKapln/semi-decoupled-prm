"""
Visualization utilities for MultiRobotFlowSolver.

This module provides functions to visualize:
- Grid decomposition
- Robot paths through the grid
- Cell occupancy over time
- Flow patterns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Tuple

from discopygal.bindings import Point_2
from multi_robot_flow_solver import MultiRobotFlowSolver, GridCell


def plot_grid_decomposition(solver: MultiRobotFlowSolver, ax=None):
    """
    Visualize the grid decomposition with cell boundaries.

    Args:
        solver: The MultiRobotFlowSolver instance (after load_scene)
        ax: Matplotlib axes (creates new if None)

    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid cells
    for cell_id, cell in solver.grid_cells.items():
        rect = patches.Rectangle(
            (cell.min_x, cell.min_y),
            cell.max_x - cell.min_x,
            cell.max_y - cell.min_y,
            linewidth=1,
            edgecolor='gray',
            facecolor='none',
            alpha=0.5
        )
        ax.add_patch(rect)

        # Add cell ID label
        center_x = (cell.min_x + cell.max_x) / 2
        center_y = (cell.min_y + cell.max_y) / 2
        ax.text(center_x, center_y, str(cell_id),
                ha='center', va='center', fontsize=8, color='gray')

    # Draw obstacles
    for obstacle in solver.scene.obstacles:
        obs_points = [(p.x().to_double(), p.y().to_double())
                      for p in obstacle.poly.vertices()]
        obs_points.append(obs_points[0])  # Close the polygon
        xs, ys = zip(*obs_points)
        ax.fill(xs, ys, color='red', alpha=0.3, label='Obstacle')

    # Draw robot start/end positions
    for robot_idx, robot in enumerate(solver.scene.robots):
        start_x = robot.start.x().to_double()
        start_y = robot.start.y().to_double()
        end_x = robot.end.x().to_double()
        end_y = robot.end.y().to_double()

        # Start position (circle)
        ax.plot(start_x, start_y, 'go', markersize=10,
                label=f'Robot {robot_idx} Start' if robot_idx == 0 else '')
        ax.text(start_x, start_y + 0.5, f'S{robot_idx}',
                ha='center', fontsize=9, fontweight='bold')

        # End position (star)
        ax.plot(end_x, end_y, 'r*', markersize=15,
                label=f'Robot {robot_idx} Goal' if robot_idx == 0 else '')
        ax.text(end_x, end_y + 0.5, f'G{robot_idx}',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Grid Decomposition')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_paths(solver: MultiRobotFlowSolver, path_collection, ax=None):
    """
    Visualize robot paths on the grid.

    Args:
        solver: The MultiRobotFlowSolver instance
        path_collection: PathCollection with solution paths
        ax: Matplotlib axes (creates new if None)

    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    # First draw the grid
    plot_grid_decomposition(solver, ax)

    # Define colors for different robots
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

    # Draw paths
    for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
        color = colors[robot_idx % len(colors)]

        # Extract path coordinates
        path_x = []
        path_y = []
        for point in path.points:
            x = point.location.x().to_double() if hasattr(point.location.x(), 'to_double') else float(point.location.x())
            y = point.location.y().to_double() if hasattr(point.location.y(), 'to_double') else float(point.location.y())
            path_x.append(x)
            path_y.append(y)

        # Plot path as line
        ax.plot(path_x, path_y, '-', color=color, linewidth=2,
                alpha=0.7, label=f'Robot {robot_idx}')

        # Mark waypoints
        ax.plot(path_x[1:-1], path_y[1:-1], 'o', color=color,
                markersize=5, alpha=0.6)

    ax.set_title('Robot Paths Through Grid')
    ax.legend()

    return ax


def plot_cell_occupancy_heatmap(solver: MultiRobotFlowSolver,
                                 path_collection,
                                 timestep: int,
                                 ax=None):
    """
    Create a heatmap showing cell occupancy at a specific timestep.

    Args:
        solver: The MultiRobotFlowSolver instance
        path_collection: PathCollection with solution paths
        timestep: Which timestep to visualize
        ax: Matplotlib axes (creates new if None)

    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Create occupancy grid
    occupancy = np.zeros((solver.num_rows, solver.num_cols))

    # Count robots in each cell at this timestep
    for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
        if timestep < len(path.points):
            point = path.points[timestep].location
            cell_id = solver._find_cell_for_point(point)
            if cell_id != -1:
                cell = solver.grid_cells[cell_id]
                occupancy[cell.row, cell.col] += 1

    # Create heatmap
    im = ax.imshow(occupancy, cmap='YlOrRd', origin='lower',
                   aspect='auto', vmin=0, vmax=solver.max_density + 1)

    # Add grid lines
    for i in range(solver.num_rows + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(solver.num_cols + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    # Add cell values
    for i in range(solver.num_rows):
        for j in range(solver.num_cols):
            text = ax.text(j, i, int(occupancy[i, j]),
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'Cell Occupancy at Timestep {timestep}')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Number of Robots')

    return ax


def create_animation(solver: MultiRobotFlowSolver,
                     path_collection,
                     filename='robot_paths.gif',
                     fps=2):
    """
    Create an animated visualization of robots moving through the grid.

    Args:
        solver: The MultiRobotFlowSolver instance
        path_collection: PathCollection with solution paths
        filename: Output filename for animation
        fps: Frames per second

    Returns:
        matplotlib animation object
    """
    # Find max timesteps
    max_timesteps = max(len(path.points) for path in path_collection.paths.values())

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Colors for robots
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

    # Robot positions at each timestep
    robot_trajectories = {}
    for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
        trajectory = []
        for point in path.points:
            x = point.location.x().to_double() if hasattr(point.location.x(), 'to_double') else float(point.location.x())
            y = point.location.y().to_double() if hasattr(point.location.y(), 'to_double') else float(point.location.y())
            trajectory.append((x, y))
        robot_trajectories[robot_idx] = trajectory

    def animate(frame):
        ax1.clear()
        ax2.clear()

        # Left plot: Current positions and trails
        plot_grid_decomposition(solver, ax1)

        for robot_idx, trajectory in robot_trajectories.items():
            color = colors[robot_idx % len(colors)]

            # Draw trail up to current frame
            trail_length = min(frame + 1, len(trajectory))
            trail_x = [trajectory[i][0] for i in range(trail_length)]
            trail_y = [trajectory[i][1] for i in range(trail_length)]

            if trail_length > 0:
                ax1.plot(trail_x, trail_y, '-', color=color,
                        linewidth=2, alpha=0.5)

                # Current position
                current_x, current_y = trajectory[min(frame, len(trajectory) - 1)]
                ax1.plot(current_x, current_y, 'o', color=color,
                        markersize=12, label=f'Robot {robot_idx}')

        ax1.set_title(f'Robot Positions at Timestep {frame}')
        ax1.legend()

        # Right plot: Occupancy heatmap
        plot_cell_occupancy_heatmap(solver, path_collection, frame, ax2)

        return ax1, ax2

    # Create animation
    anim = FuncAnimation(fig, animate, frames=max_timesteps,
                        interval=1000/fps, blit=False)

    # Save animation
    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying animation instead...")
        plt.show()

    return anim


def analyze_flow_statistics(solver: MultiRobotFlowSolver, path_collection):
    """
    Compute and print statistics about the flow solution.

    Args:
        solver: The MultiRobotFlowSolver instance
        path_collection: PathCollection with solution paths

    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_robots': len(path_collection.paths),
        'num_cells': len(solver.grid_cells),
        'cell_size': solver.cell_size,
        'max_density': solver.max_density,
        'path_lengths': [],
        'max_timesteps': 0,
        'total_cell_visits': 0,
        'max_cell_occupancy': 0,
        'avg_cell_occupancy': 0
    }

    # Compute path statistics
    for robot, path in path_collection.paths.items():
        path_length = len(path.points)
        stats['path_lengths'].append(path_length)
        stats['max_timesteps'] = max(stats['max_timesteps'], path_length)

    stats['avg_path_length'] = np.mean(stats['path_lengths'])
    stats['min_path_length'] = min(stats['path_lengths'])
    stats['max_path_length'] = max(stats['path_lengths'])

    # Compute cell occupancy statistics
    cell_visit_count = {cell_id: 0 for cell_id in solver.grid_cells.keys()}

    for timestep in range(stats['max_timesteps']):
        occupancy_at_t = {}
        for robot, path in path_collection.paths.items():
            if timestep < len(path.points):
                point = path.points[timestep].location
                cell_id = solver._find_cell_for_point(point)
                if cell_id != -1:
                    occupancy_at_t[cell_id] = occupancy_at_t.get(cell_id, 0) + 1
                    cell_visit_count[cell_id] += 1

        if occupancy_at_t:
            max_at_t = max(occupancy_at_t.values())
            stats['max_cell_occupancy'] = max(stats['max_cell_occupancy'], max_at_t)

    stats['total_cell_visits'] = sum(cell_visit_count.values())
    stats['avg_cell_occupancy'] = stats['total_cell_visits'] / len(solver.grid_cells)

    # Print statistics
    print("\n" + "=" * 60)
    print("FLOW SOLUTION STATISTICS")
    print("=" * 60)
    print(f"Number of robots: {stats['num_robots']}")
    print(f"Grid size: {solver.num_rows} × {solver.num_cols} = {stats['num_cells']} cells")
    print(f"Cell size: {stats['cell_size']}")
    print(f"Max density: {stats['max_density']}")
    print(f"\nPath Statistics:")
    print(f"  Average path length: {stats['avg_path_length']:.2f} steps")
    print(f"  Min path length: {stats['min_path_length']} steps")
    print(f"  Max path length: {stats['max_path_length']} steps")
    print(f"  Total timesteps: {stats['max_timesteps']}")
    print(f"\nCell Occupancy:")
    print(f"  Total cell-visits: {stats['total_cell_visits']}")
    print(f"  Average visits per cell: {stats['avg_cell_occupancy']:.2f}")
    print(f"  Max simultaneous occupancy: {stats['max_cell_occupancy']}")
    print(f"  Density constraint: {'SATISFIED' if stats['max_cell_occupancy'] <= stats['max_density'] else 'VIOLATED'}")
    print("=" * 60)

    return stats


def visualize_complete_solution(solver: MultiRobotFlowSolver,
                                path_collection,
                                save_prefix='solution'):
    """
    Create a complete visualization with multiple views.

    Args:
        solver: The MultiRobotFlowSolver instance
        path_collection: PathCollection with solution paths
        save_prefix: Prefix for saved figure files
    """
    # Figure 1: Grid decomposition
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    plot_grid_decomposition(solver, ax1)
    plt.tight_layout()
    fig1.savefig(f'{save_prefix}_grid.png', dpi=150)
    print(f"Saved {save_prefix}_grid.png")

    # Figure 2: Paths
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    plot_paths(solver, path_collection, ax2)
    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_paths.png', dpi=150)
    print(f"Saved {save_prefix}_paths.png")

    # Figure 3: Occupancy over time (multiple subplots)
    max_timesteps = max(len(path.points) for path in path_collection.paths.values())
    num_snapshots = min(6, max_timesteps)
    timesteps = np.linspace(0, max_timesteps - 1, num_snapshots, dtype=int)

    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(timesteps):
        plot_cell_occupancy_heatmap(solver, path_collection, t, axes[i])

    plt.tight_layout()
    fig3.savefig(f'{save_prefix}_occupancy.png', dpi=150)
    print(f"Saved {save_prefix}_occupancy.png")

    # Compute and print statistics
    stats = analyze_flow_statistics(solver, path_collection)

    # Show all figures
    plt.show()

    return stats


if __name__ == '__main__':
    # Example usage
    from discopygal.solvers_infra import Scene, RobotDisc, ObstaclePolygon
    from discopygal.bindings import Point_2, FT, Polygon_2
    from multi_robot_flow_solver import MultiRobotFlowSolver

    print("Creating test scene...")
    scene = Scene()

    # Add 3 robots
    robot0 = RobotDisc(Point_2(FT(2), FT(2)), Point_2(FT(18), FT(18)), 0.3)
    robot1 = RobotDisc(Point_2(FT(18), FT(2)), Point_2(FT(2), FT(18)), 0.3)
    robot2 = RobotDisc(Point_2(FT(2), FT(10)), Point_2(FT(18), FT(10)), 0.3)
    scene.add_robot(robot0)
    scene.add_robot(robot1)
    scene.add_robot(robot2)

    # Add obstacle
    obstacle_points = [
        Point_2(FT(8), FT(8)),
        Point_2(FT(12), FT(8)),
        Point_2(FT(12), FT(12)),
        Point_2(FT(8), FT(12))
    ]
    obstacle = ObstaclePolygon(Polygon_2(obstacle_points))
    scene.add_obstacle(obstacle)

    print("Creating solver...")
    solver = MultiRobotFlowSolver(
        cell_size=5.0,
        max_density=1,
        time_horizon=20,
        flow_control_strategy='sequential',
        verbose=True
    )

    print("Solving...")
    solver.load_scene(scene)
    paths = solver.solve()

    if len(paths.paths) > 0:
        print("Creating visualizations...")
        visualize_complete_solution(solver, paths, save_prefix='example_solution')

        # Try to create animation
        print("Creating animation...")
        create_animation(solver, paths, 'example_animation.gif', fps=2)
    else:
        print("No solution found!")

