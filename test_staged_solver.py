"""Tests for the staged multi-robot solver.

Runs the full pipeline on several scenes and validates the output using
discopygal's ``verify_paths``.  Can also be loaded by solver_viewer::

    solver_viewer -sf staged_solver.py -sc scenes/scene_1.json

Test scenes
-----------
1. **scene_1.json** — 8 robots crossing in open space (no obstacles).
2. **warehouse.json** — 15 robots navigating a warehouse with 3 aisle walls.
3. **Inline 2-robot with obstacle** — simple crossing with a central box.
4. **Corridor head-on** — two robots in a narrow corridor heading at each
   other, forcing one to yield.
"""

import os
import sys
import time

from discopygal.bindings import FT, Point_2, Polygon_2
from discopygal.solvers_infra import (
    ObstaclePolygon,
    PathCollection,
    RobotDisc,
    Scene,
)
from discopygal.solvers_infra.verify_paths import verify_paths

from staged_solver import StagedSolver

SCENES_DIR = os.path.join(os.path.dirname(__file__), "scenes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_solver(
    scene: Scene,
    topology: str = "pairwise",
    strategy: str = "sequential",
    num_samples: int = 40,
    k_nearest: int = 8,
    seed: int = 42,
    time_horizon=None,
    max_cell_density: int = 4,
    verbose: bool = True,
) -> PathCollection:
    solver = StagedSolver(
        topology=topology,
        flow_strategy=strategy,
        num_samples=num_samples,
        k_nearest=k_nearest,
        time_horizon=time_horizon,
        prm_seed=seed,
        max_cell_density=max_cell_density,
    )
    solver.verbose = verbose
    solver.load_scene(scene)

    t0 = time.time()
    pc = solver.solve()
    elapsed = time.time() - t0

    print(f"  Solved in {elapsed:.2f}s")
    return pc


def _check_paths(scene: Scene, pc: PathCollection, label: str) -> bool:
    """Validate a PathCollection and print results.

    Checks two levels:
    1. **Structural** — paths exist, same length, correct start/end.
    2. **Collision-free** — discopygal ``verify_paths`` (obstacle + robot).

    All collision types (obstacle and robot-robot) are treated as failures.
    """
    if pc is None or len(pc.paths) == 0:
        print(f"  [{label}] FAIL — empty PathCollection")
        return False

    # Check all paths have the same length
    lengths = [len(p.points) for p in pc.paths.values()]
    if min(lengths) != max(lengths):
        print(f"  [{label}] FAIL — path lengths differ: {lengths}")
        return False

    # Check start/end positions match
    for robot, path in pc.paths.items():
        pts = path.points
        sx = pts[0].location.x().to_double()
        sy = pts[0].location.y().to_double()
        ex = pts[-1].location.x().to_double()
        ey = pts[-1].location.y().to_double()
        rsx = robot.start.x().to_double()
        rsy = robot.start.y().to_double()
        rex = robot.end.x().to_double()
        rey = robot.end.y().to_double()
        if abs(sx - rsx) > 1e-6 or abs(sy - rsy) > 1e-6:
            print(f"  [{label}] FAIL — start mismatch: path=({sx},{sy}) robot=({rsx},{rsy})")
            return False
        if abs(ex - rex) > 1e-6 or abs(ey - rey) > 1e-6:
            print(f"  [{label}] FAIL — end mismatch: path=({ex},{ey}) robot=({rex},{rey})")
            return False

    # Use discopygal's verify_paths for collision checking
    valid, reason = verify_paths(scene, pc)
    if valid:
        print(f"  [{label}] PASS — {len(pc.paths)} robots, "
              f"{lengths[0]} waypoints, verify_paths OK")
    else:
        print(f"  [{label}] FAIL — verify_paths: {reason}")
    return valid


# ---------------------------------------------------------------------------
# Inline test scenes
# ---------------------------------------------------------------------------

def create_crossing_scene():
    """Two robots crossing with a central obstacle."""
    scene = Scene()
    scene.add_robot(RobotDisc(
        start=Point_2(FT(1), FT(1)),
        end=Point_2(FT(9), FT(9)),
        radius=FT(0.5),
    ))
    scene.add_robot(RobotDisc(
        start=Point_2(FT(9), FT(1)),
        end=Point_2(FT(1), FT(9)),
        radius=FT(0.5),
    ))
    scene.add_obstacle(ObstaclePolygon(Polygon_2([
        Point_2(FT(4), FT(4)), Point_2(FT(6), FT(4)),
        Point_2(FT(6), FT(6)), Point_2(FT(4), FT(6)),
    ])))
    return scene


def create_corridor_scene():
    """Narrow corridor with two robots heading at each other.

    Layout (top view, not to scale)::

        +-----------------------------+
        |  wall                       |
        +---+                   +-----+
        | S1|      corridor     | S2  |
        +---+                   +-----+
        |  wall                       |
        +-----------------------------+

    Robot 1 starts at the left end, goal at the right.
    Robot 2 starts at the right end, goal at the left.
    The corridor is wide enough for both robots to pass with care.
    """
    scene = Scene()
    r = 0.3
    scene.add_robot(RobotDisc(
        start=Point_2(FT(1), FT(5)),
        end=Point_2(FT(9), FT(5)),
        radius=FT(r),
    ))
    scene.add_robot(RobotDisc(
        start=Point_2(FT(9), FT(5)),
        end=Point_2(FT(1), FT(5)),
        radius=FT(r),
    ))
    # Top wall
    scene.add_obstacle(ObstaclePolygon(Polygon_2([
        Point_2(FT(0), FT(6.5)), Point_2(FT(10), FT(6.5)),
        Point_2(FT(10), FT(10)), Point_2(FT(0), FT(10)),
    ])))
    # Bottom wall
    scene.add_obstacle(ObstaclePolygon(Polygon_2([
        Point_2(FT(0), FT(0)), Point_2(FT(10), FT(0)),
        Point_2(FT(10), FT(3.5)), Point_2(FT(0), FT(3.5)),
    ])))
    return scene


def create_three_robot_scene():
    """Three robots around a central obstacle.

    One pair has crossing paths, the third goes around.
    """
    scene = Scene()
    r = 0.4
    scene.add_robot(RobotDisc(
        start=Point_2(FT(1), FT(1)),
        end=Point_2(FT(9), FT(9)),
        radius=FT(r),
    ))
    scene.add_robot(RobotDisc(
        start=Point_2(FT(9), FT(1)),
        end=Point_2(FT(1), FT(9)),
        radius=FT(r),
    ))
    scene.add_robot(RobotDisc(
        start=Point_2(FT(5), FT(1)),
        end=Point_2(FT(5), FT(9)),
        radius=FT(r),
    ))
    scene.add_obstacle(ObstaclePolygon(Polygon_2([
        Point_2(FT(4), FT(4)), Point_2(FT(6), FT(4)),
        Point_2(FT(6), FT(6)), Point_2(FT(4), FT(6)),
    ])))
    return scene


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_crossing():
    """2 robots crossing with central obstacle."""
    print("\n=== Test: 2-robot crossing with obstacle ===")
    scene = create_crossing_scene()
    for topo in ("pairwise", "star"):
        print(f"\n  topology={topo}")
        pc = _run_solver(scene, topology=topo)
        _check_paths(scene, pc, f"crossing/{topo}")


def test_corridor():
    """2 robots head-on in a narrow corridor."""
    print("\n=== Test: corridor head-on ===")
    scene = create_corridor_scene()
    pc = _run_solver(scene, num_samples=50, time_horizon=20)
    _check_paths(scene, pc, "corridor")


def test_three_robots():
    """3 robots around obstacle."""
    print("\n=== Test: 3 robots with obstacle ===")
    scene = create_three_robot_scene()
    pc = _run_solver(scene, strategy="priority", num_samples=40)
    _check_paths(scene, pc, "3-robot")


def test_scene_1():
    """8 robots crossing (no obstacles) from scenes/scene_1.json."""
    print("\n=== Test: scene_1.json (8 robots, no obstacles) ===")
    path = os.path.join(SCENES_DIR, "scene_1.json")
    if not os.path.exists(path):
        print(f"  SKIP — {path} not found")
        return
    scene = Scene.from_file(path)
    print(f"  {len(scene.robots)} robots, {len(scene.obstacles)} obstacles")
    pc = _run_solver(scene, num_samples=30, time_horizon=30)
    _check_paths(scene, pc, "scene_1")


def test_warehouse():
    """15 robots in a warehouse with aisles from scenes/warehouse.json."""
    print("\n=== Test: warehouse.json (15 robots, 7 obstacles) ===")
    path = os.path.join(SCENES_DIR, "warehouse.json")
    if not os.path.exists(path):
        print(f"  SKIP — {path} not found")
        return
    scene = Scene.from_file(path)
    print(f"  {len(scene.robots)} robots, {len(scene.obstacles)} obstacles")
    pc = _run_solver(scene, num_samples=50, time_horizon=40, max_cell_density=1000)
    _check_paths(scene, pc, "warehouse")


def test_solver_viewer_compat():
    """Verify that StagedSolver exposes graph and arrangement for solver_viewer."""
    print("\n=== Test: solver_viewer compatibility ===")
    scene = create_crossing_scene()
    solver = StagedSolver(num_samples=20, prm_seed=42)
    solver.verbose = False
    solver.load_scene(scene)
    pc = solver.solve()

    graph = solver.get_graph()
    arr = solver.get_arrangement()

    ok = True
    if graph is None:
        print("  FAIL — get_graph() returned None")
        ok = False
    else:
        print(f"  get_graph(): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if arr is None:
        print("  FAIL — get_arrangement() returned None")
        ok = False
    else:
        n_faces = sum(1 for _ in arr.faces())
        print(f"  get_arrangement(): {n_faces} faces")

    if ok:
        print("  [viewer_compat] PASS")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Staged Solver Test Suite")
    print("=" * 60)

    test_crossing()
    test_corridor()
    test_three_robots()
    test_scene_1()
    test_warehouse()
    test_solver_viewer_compat()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
