"""End-to-end tests for StagedSolver. Outputs are validated via
verify_paths."""

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
    num_samples: int = 40,
    k_nearest: int = 8,
    max_cell_density: int = 100,
    verbose: bool = True,
) -> PathCollection:
    solver = StagedSolver(
        num_samples=num_samples,
        k_nearest=k_nearest,
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
    """Structural check (length + endpoints) + verify_paths."""
    if pc is None or len(pc.paths) == 0:
        print(f"  [{label}] FAIL: empty PathCollection")
        return False

    lengths = [len(p.points) for p in pc.paths.values()]
    if min(lengths) != max(lengths):
        print(f"  [{label}] FAIL: path lengths differ: {lengths}")
        return False

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
            print(f"  [{label}] FAIL: start mismatch: path=({sx},{sy}) robot=({rsx},{rsy})")
            return False
        if abs(ex - rex) > 1e-6 or abs(ey - rey) > 1e-6:
            print(f"  [{label}] FAIL: end mismatch: path=({ex},{ey}) robot=({rex},{rey})")
            return False

    valid, reason = verify_paths(scene, pc)
    if valid:
        print(f"  [{label}] PASS: {len(pc.paths)} robots, "
              f"{lengths[0]} waypoints, verify_paths OK")
    else:
        print(f"  [{label}] FAIL: verify_paths: {reason}")
    return valid


# ---------------------------------------------------------------------------
# Inline test scenes
# ---------------------------------------------------------------------------

def create_crossing_scene():
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
    """Horizontal corridor, head-on. Free strip y in [3.5, 6.5]; robots
    can pass side-by-side (sanity check, not a forced yield)."""
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
    """Three robots around a central obstacle (one pair crosses)."""
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
    print("\n=== Test: 2-robot crossing with obstacle ===")
    scene = create_crossing_scene()
    pc = _run_solver(scene)
    _check_paths(scene, pc, "crossing")


def test_corridor():
    print("\n=== Test: corridor head-on ===")
    scene = create_corridor_scene()
    pc = _run_solver(scene, num_samples=50)
    _check_paths(scene, pc, "corridor")


def test_three_robots():
    print("\n=== Test: 3 robots with obstacle ===")
    scene = create_three_robot_scene()
    pc = _run_solver(scene, num_samples=40)
    _check_paths(scene, pc, "3-robot")


def test_scene_1():
    print("\n=== Test: scene_1.json (8 robots, no obstacles) ===")
    path = os.path.join(SCENES_DIR, "scene_1.json")
    if not os.path.exists(path):
        print(f"  SKIP:{path} not found")
        return
    scene = Scene.from_file(path)
    scene._source_path = path
    print(f"  {len(scene.robots)} robots, {len(scene.obstacles)} obstacles")
    pc = _run_solver(scene, num_samples=30)
    _check_paths(scene, pc, "scene_1")


def test_warehouse():
    print("\n=== Test: warehouse.json (15 robots, 7 obstacles) ===")
    path = os.path.join(SCENES_DIR, "warehouse.json")
    if not os.path.exists(path):
        print(f"  SKIP:{path} not found")
        return
    scene = Scene.from_file(path)
    scene._source_path = path
    print(f"  {len(scene.robots)} robots, {len(scene.obstacles)} obstacles")
    pc = _run_solver(scene, num_samples=50)
    _check_paths(scene, pc, "warehouse")


def test_solver_viewer_compat():
    print("\n=== Test: solver_viewer compatibility ===")
    scene = create_crossing_scene()
    solver = StagedSolver(num_samples=20)
    solver.verbose = False
    solver.load_scene(scene)
    pc = solver.solve()

    graph = solver.get_graph()
    arr = solver.get_arrangement()

    ok = True
    if graph is None:
        print("  FAIL:get_graph() returned None")
        ok = False
    else:
        print(f"  get_graph(): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if arr is None:
        print("  FAIL:get_arrangement() returned None")
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
