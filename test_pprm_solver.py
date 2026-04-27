import os
import time

from discopygal.solvers_infra import (
    PathCollection,
    Scene,
)
from discopygal.solvers_infra.verify_paths import verify_paths

from pprm_solver import pPRMSolver

SCENES_DIR = os.path.join(os.path.dirname(__file__), "scenes")

# Helpers

def _run_solver(
    scene: Scene,
    num_samples: int = 40,
    k_nearest: int = 8,
    max_cell_density: int = 100,
    verbose: bool = True,
) -> PathCollection:
    solver = pPRMSolver(
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
    path = os.path.join(SCENES_DIR, "scene_1.json")
    if not os.path.exists(path):
        print(f"  SKIP:{path} not found")
        return
    scene = Scene.from_file(path)
    scene._source_path = path
    solver = pPRMSolver(num_samples=20)
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



if __name__ == "__main__":
    test_scene_1()
    test_warehouse()
    test_solver_viewer_compat()