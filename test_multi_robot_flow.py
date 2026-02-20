"""
Test and demonstration script for MultiRobotFlowSolver.

This script shows how to:
1. Create a scene with multiple robots
2. Use the MultiRobotFlowSolver with different strategies
3. Visualize the results
"""

from discopygal.bindings import Point_2, FT, Polygon_2
from discopygal.solvers_infra import Scene, RobotDisc, ObstaclePolygon
from multi_robot_flow_solver import MultiRobotFlowSolver, visualize_grid, print_grid_info


def create_simple_test_scene():
    """
    Create a simple test scene with 3 robots and a few obstacles.

    Workspace: 20x20 grid
    Robots: 3 disc robots with radius 0.3
    Obstacles: 2 rectangular obstacles
    """
    # Create scene
    scene = Scene()

    # Add 3 robots with start/end positions
    # Robot 0: bottom-left to top-right
    robot0 = RobotDisc(
        start=Point_2(FT(1), FT(1)),
        end=Point_2(FT(18), FT(18)),
        radius=0.3
    )
    scene.add_robot(robot0)

    # Robot 1: bottom-right to top-left
    robot1 = RobotDisc(
        start=Point_2(FT(18), FT(1)),
        end=Point_2(FT(1), FT(18)),
        radius=0.3
    )
    scene.add_robot(robot1)

    # Robot 2: left-middle to right-middle
    robot2 = RobotDisc(
        start=Point_2(FT(1), FT(10)),
        end=Point_2(FT(18), FT(10)),
        radius=0.3
    )
    scene.add_robot(robot2)

    # Add obstacles
    # Obstacle 1: rectangle in upper-left quadrant
    obstacle1_points = [
        Point_2(FT(5), FT(12)),
        Point_2(FT(8), FT(12)),
        Point_2(FT(8), FT(15)),
        Point_2(FT(5), FT(15))
    ]
    obstacle1 = ObstaclePolygon(Polygon_2(obstacle1_points))
    scene.add_obstacle(obstacle1)

    # Obstacle 2: rectangle in lower-right quadrant
    obstacle2_points = [
        Point_2(FT(12), FT(5)),
        Point_2(FT(15), FT(5)),
        Point_2(FT(15), FT(8)),
        Point_2(FT(12), FT(8))
    ]
    obstacle2 = ObstaclePolygon(Polygon_2(obstacle2_points))
    scene.add_obstacle(obstacle2)

    return scene


def create_corridor_test_scene():
    """
    Create a corridor scenario where robots need to coordinate.

    Two robots need to pass each other in a narrow corridor.
    """
    scene = Scene()

    # Robot 0: left to right
    robot0 = RobotDisc(
        start=Point_2(FT(1), FT(5)),
        end=Point_2(FT(18), FT(5)),
        radius=0.3
    )
    scene.add_robot(robot0)

    # Robot 1: right to left
    robot1 = RobotDisc(
        start=Point_2(FT(18), FT(5)),
        end=Point_2(FT(1), FT(5)),
        radius=0.3
    )
    scene.add_robot(robot1)

    # Create corridor walls (two horizontal obstacles above and below)
    # Upper wall
    upper_wall_points = [
        Point_2(FT(0), FT(8)),
        Point_2(FT(20), FT(8)),
        Point_2(FT(20), FT(10)),
        Point_2(FT(0), FT(10))
    ]
    upper_wall = ObstaclePolygon(Polygon_2(upper_wall_points))
    scene.add_obstacle(upper_wall)

    # Lower wall
    lower_wall_points = [
        Point_2(FT(0), FT(0)),
        Point_2(FT(20), FT(0)),
        Point_2(FT(20), FT(2)),
        Point_2(FT(0), FT(2))
    ]
    lower_wall = ObstaclePolygon(Polygon_2(lower_wall_points))
    scene.add_obstacle(lower_wall)

    return scene


def test_ilp_strategy():
    """Test the ILP-based multi-commodity flow strategy."""
    print("=" * 60)
    print("Testing ILP Strategy")
    print("=" * 60)

    scene = create_simple_test_scene()

    # Create solver with ILP strategy
    solver = MultiRobotFlowSolver(
        cell_size=5.0,
        max_density=1,  # Only 1 robot per cell at a time
        time_horizon=20,
        flow_control_strategy='ilp',
        verbose=True
    )

    # Load scene and solve
    solver.load_scene(scene)

    # Print grid info
    print_grid_info(solver)

    # Solve
    print("\nSolving...")
    path_collection = solver.solve()

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    if len(path_collection.paths) > 0:
        print(f"✓ Found paths for {len(path_collection.paths)} robots")
        for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
            print(f"\nRobot {robot_idx}:")
            print(f"  Path length: {len(path.points)} waypoints")
            print(f"  Waypoints:")
            for i, point in enumerate(path.points):
                loc = point.location
                x = loc.x().to_double() if hasattr(loc.x(), 'to_double') else float(loc.x())
                y = loc.y().to_double() if hasattr(loc.y(), 'to_double') else float(loc.y())
                print(f"    {i}: ({x:.2f}, {y:.2f})")
    else:
        print("✗ No solution found")

    return path_collection


def test_sequential_strategy():
    """Test the sequential planning strategy."""
    print("\n" + "=" * 60)
    print("Testing Sequential Strategy")
    print("=" * 60)

    scene = create_simple_test_scene()

    # Create solver with sequential strategy
    solver = MultiRobotFlowSolver(
        cell_size=5.0,
        max_density=1,
        time_horizon=30,
        flow_control_strategy='sequential',
        verbose=True
    )

    solver.load_scene(scene)

    print("\nSolving...")
    path_collection = solver.solve()

    print("\n" + "=" * 60)
    print("Results:")
    if len(path_collection.paths) > 0:
        print(f"✓ Found paths for {len(path_collection.paths)} robots")
        for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
            print(f"  Robot {robot_idx}: {len(path.points)} waypoints")
    else:
        print("✗ No solution found")

    return path_collection


def test_priority_strategy():
    """Test the priority-based planning strategy."""
    print("\n" + "=" * 60)
    print("Testing Priority-Based Strategy")
    print("=" * 60)

    scene = create_simple_test_scene()

    # Create solver with priority strategy
    solver = MultiRobotFlowSolver(
        cell_size=5.0,
        max_density=1,
        time_horizon=30,
        flow_control_strategy='priority',
        verbose=True
    )

    solver.load_scene(scene)

    print("\nSolving...")
    path_collection = solver.solve()

    print("\n" + "=" * 60)
    print("Results:")
    if len(path_collection.paths) > 0:
        print(f"✓ Found paths for {len(path_collection.paths)} robots")
        for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
            print(f"  Robot {robot_idx}: {len(path.points)} waypoints")
    else:
        print("✗ No solution found")

    return path_collection


def test_corridor_scenario():
    """Test corridor scenario where robots must coordinate."""
    print("\n" + "=" * 60)
    print("Testing Corridor Scenario (Sequential)")
    print("=" * 60)

    scene = create_corridor_test_scene()

    solver = MultiRobotFlowSolver(
        cell_size=3.0,  # Smaller cells for corridor
        max_density=1,
        time_horizon=20,
        flow_control_strategy='sequential',
        verbose=True
    )

    solver.load_scene(scene)
    print_grid_info(solver)

    print("\nSolving...")
    path_collection = solver.solve()

    print("\n" + "=" * 60)
    print("Results:")
    if len(path_collection.paths) > 0:
        print(f"✓ Found paths for {len(path_collection.paths)} robots")
        for robot_idx, (robot, path) in enumerate(path_collection.paths.items()):
            print(f"  Robot {robot_idx}: {len(path.points)} waypoints")
    else:
        print("✗ No solution found")

    return path_collection


def compare_strategies():
    """Compare all three strategies on the same scene."""
    print("\n" + "=" * 60)
    print("Comparing All Strategies")
    print("=" * 60)

    scene = create_simple_test_scene()
    strategies = ['sequential', 'priority']

    # Add ILP only if OR-Tools is available
    try:
        from ortools.linear_solver import pywraplp
        strategies.insert(0, 'ilp')
    except ImportError:
        print("Note: OR-Tools not available, skipping ILP strategy\n")

    results = {}

    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} strategy ---")

        solver = MultiRobotFlowSolver(
            cell_size=5.0,
            max_density=1,
            time_horizon=25,
            flow_control_strategy=strategy,
            verbose=False
        )

        solver.load_scene(scene)
        path_collection = solver.solve()

        if len(path_collection.paths) > 0:
            # Compute total path length
            total_length = 0
            for robot, path in path_collection.paths.items():
                total_length += len(path.points)

            results[strategy] = {
                'success': True,
                'num_robots': len(path_collection.paths),
                'total_waypoints': total_length,
                'avg_waypoints': total_length / len(path_collection.paths)
            }
            print(f"  ✓ Success: {len(path_collection.paths)} robots, "
                  f"{total_length} total waypoints, "
                  f"{results[strategy]['avg_waypoints']:.1f} avg per robot")
        else:
            results[strategy] = {'success': False}
            print(f"  ✗ Failed to find solution")

    print("\n" + "=" * 60)
    print("Summary:")
    for strategy, result in results.items():
        if result['success']:
            print(f"  {strategy:12s}: ✓ (avg waypoints: {result['avg_waypoints']:.1f})")
        else:
            print(f"  {strategy:12s}: ✗")

    return results


if __name__ == '__main__':
    # Run tests
    print("Multi-Robot Flow Solver Test Suite\n")

    # Test individual strategies
    try:
        test_ilp_strategy()
    except Exception as e:
        print(f"ILP test failed: {e}")

    try:
        test_sequential_strategy()
    except Exception as e:
        print(f"Sequential test failed: {e}")

    try:
        test_priority_strategy()
    except Exception as e:
        print(f"Priority test failed: {e}")

    try:
        test_corridor_scenario()
    except Exception as e:
        print(f"Corridor test failed: {e}")

    # Compare strategies
    try:
        compare_strategies()
    except Exception as e:
        print(f"Comparison failed: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

