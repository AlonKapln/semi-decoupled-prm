"""Benchmark the staged solver against discopygal's own solvers.

For each (scene, solver) pair, runs the solver in a fresh subprocess
with a wall-clock timeout (default 5 minutes), validates the resulting
``PathCollection`` with ``verify_paths``, and writes one row to a CSV.

Usage::

    python3 compare_solvers.py                          # all scenes, 5 min cap
    python3 compare_solvers.py --timeout 120            # 2 min cap
    python3 compare_solvers.py --scenes swap_2 rotate_6 # subset
    python3 compare_solvers.py --solvers Staged PRM     # subset
    python3 compare_solvers.py --out my_results.csv

CSV columns::

    scene, solver, num_robots, num_obstacles, status, elapsed_sec,
    num_paths, max_waypoints, verify_ok, notes

``status`` is one of ``ok``, ``invalid`` (paths returned but
verify_paths rejected), ``empty`` (no paths), ``timeout``, ``crash``.

Each run goes through a spawn-context subprocess so the timeout is
enforced by terminating the child, and one solver's crash can't take
the whole benchmark down.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")


# ---------------------------------------------------------------------------
# Solver factories
# ---------------------------------------------------------------------------
# Hyperparameters are chosen so each solver has a realistic chance on
# multi-robot scenes without blowing past the timeout setup time.  PRM /
# dRRT land counts are high enough to cover the joint configuration
# spaces of our small/medium scenes; on the warehouse-class scenes they
# will typically time out, which is a legitimate benchmark outcome.

def _make_staged():
    from staged_solver import StagedSolver
    return StagedSolver(
        num_samples=40, k_nearest=8, prm_seed=42,
        max_cell_density=100,
    )


def _make_prm():
    from discopygal.solvers.prm import PRM
    return PRM(num_landmarks=3000, k_nn=15)


def _make_rrt():
    from discopygal.solvers.rrt import RRT
    return RRT(num_landmarks=5000, eta=0.5)


def _make_birrt():
    from discopygal.solvers.rrt import BiRRT
    # BiRRT inherits RRT's __init__ kwargs (num_landmarks, eta).
    return BiRRT(n_join=5, num_landmarks=5000, eta=0.5)


def _make_drrt():
    from discopygal.solvers.rrt import dRRT
    return dRRT(num_landmarks=1500, k_nn=15, prm_num_landmarks=1500)


def _make_drrt_star():
    from discopygal.solvers.rrt import dRRT_star
    # dRRT_star inherits dRRT's __init__ kwargs (num_landmarks, k_nn).
    return dRRT_star(
        num_expands=5000, random_sample_counter=10,
        num_landmarks=1500, k_nn=15,
    )


def _make_staggered_grid():
    from discopygal.solvers.staggered_grid import StaggeredGrid
    # Inherits StaggeredGridBase kwargs (eps, delta). eps is a clearance
    # tolerance; delta controls grid step relative to robot radius.
    return StaggeredGrid(eps=0.01, delta=0.1)


def _make_exact_single():
    from discopygal.solvers.exact import ExactSingle
    return ExactSingle(eps=0.1)


SOLVERS: Dict[str, Callable[[], Any]] = {
    "Staged":        _make_staged,
    "PRM":           _make_prm,
    "RRT":           _make_rrt,
    "BiRRT":         _make_birrt,
    "dRRT":          _make_drrt,
    "dRRT_star":     _make_drrt_star,
    "StaggeredGrid": _make_staggered_grid,
    "ExactSingle":   _make_exact_single,
}


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

# Ordered so the benchmark progresses from cheap/easy to expensive/hard.
_DEFAULT_SCENE_ORDER = [
    "single_robot_empty",
    "L_corridor_single",
    "maze_single",
    "swap_2",
    "narrow_corridor",
    "crossing_4",
    "bottleneck_funnel",
    "spiral_4",
    "rotate_6",
    "open_16",
    "scene_1",
    "warehouse",
    "tight_rooms",
    "extreme_warehouse",
    "warehouse_rooms",
    "new_warhouse",
    "unsolvable_trap",
]


def discover_scenes(selected: Optional[List[str]]) -> List[str]:
    available = {
        os.path.splitext(f)[0]: os.path.join(SCENES_DIR, f)
        for f in os.listdir(SCENES_DIR)
        if f.endswith(".json")
    }
    if selected:
        missing = [s for s in selected if s not in available]
        if missing:
            raise SystemExit(f"Unknown scenes: {missing}")
        return [available[s] for s in selected]
    # Ordered: known-order first, then whatever's left alphabetically.
    known = [available[n] for n in _DEFAULT_SCENE_ORDER if n in available]
    leftover = sorted(
        available[n] for n in available if n not in _DEFAULT_SCENE_ORDER
    )
    return known + leftover


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

def _child(scene_path: str, solver_name: str, out_path: str) -> None:
    """Runs in a spawn-context subprocess.  Writes one JSON blob."""
    result: Dict[str, Any] = {}
    try:
        from discopygal.solvers_infra import Scene
        from discopygal.solvers_infra.verify_paths import verify_paths
        scene = Scene.from_file(scene_path)
        factory = SOLVERS[solver_name]
        solver = factory()
        try:
            solver.disable_verbose()
        except Exception:
            pass
        solver.load_scene(scene)
        t0 = time.time()
        pc = solver.solve()
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 3)
        if pc is None or len(pc.paths) == 0:
            result["status"] = "empty"
            result["num_paths"] = 0
        else:
            result["num_paths"] = len(pc.paths)
            lens = [len(p.points) for p in pc.paths.values()]
            result["max_waypoints"] = max(lens)
            try:
                ok, reason = verify_paths(scene, pc)
                result["verify_ok"] = bool(ok)
                result["status"] = "ok" if ok else "invalid"
                result["notes"] = str(reason)[:300] if reason else ""
            except Exception as e:
                result["verify_ok"] = False
                result["status"] = "invalid"
                result["notes"] = f"verify_paths error: {e}"[:300]
    except Exception as e:
        result["status"] = "crash"
        result["notes"] = (
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )[:600]
    with open(out_path, "w") as f:
        json.dump(result, f)


def run_one(
        scene_path: str, solver_name: str, timeout: float,
) -> Dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_child,
        args=(scene_path, solver_name, tmp.name),
    )
    wall_start = time.time()
    proc.start()
    proc.join(timeout)
    wall = time.time() - wall_start
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        os.unlink(tmp.name)
        return {
            "status": "timeout",
            "elapsed_sec": round(wall, 3),
            "notes": f"killed after {timeout}s",
        }
    try:
        with open(tmp.name) as f:
            data = json.load(f)
    except Exception as e:
        data = {"status": "crash", "notes": f"no result file: {e}"}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return data


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------

def scene_metadata(path: str) -> Dict[str, int]:
    with open(path) as f:
        d = json.load(f)
    return {
        "num_robots": len(d.get("robots", [])),
        "num_obstacles": len(d.get("obstacles", [])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "scene", "solver", "num_robots", "num_obstacles", "status",
    "elapsed_sec", "num_paths", "max_waypoints", "verify_ok", "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="Per-run wall-clock cap in seconds (default: 300)")
    ap.add_argument("--out", default="benchmark_results.csv",
                    help="Output CSV path")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Scene basenames (no .json). Default: all.")
    ap.add_argument("--solvers", nargs="+", default=None,
                    help=f"Solver names. Default: all of {list(SOLVERS)}")
    args = ap.parse_args()

    if args.solvers:
        unknown = [s for s in args.solvers if s not in SOLVERS]
        if unknown:
            raise SystemExit(f"Unknown solvers: {unknown}")
        solvers = args.solvers
    else:
        solvers = list(SOLVERS)

    scene_paths = discover_scenes(args.scenes)
    scene_meta = {p: scene_metadata(p) for p in scene_paths}

    total = len(scene_paths) * len(solvers)
    print(f"Running {total} (scene, solver) pairs with {args.timeout:.0f}s cap each.")
    print(f"Scenes:  {[os.path.splitext(os.path.basename(p))[0] for p in scene_paths]}")
    print(f"Solvers: {solvers}")
    print(f"Output:  {args.out}")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        i = 0
        for scene_path in scene_paths:
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            meta = scene_meta[scene_path]
            for solver_name in solvers:
                i += 1
                print(
                    f"[{i:>3}/{total}] {scene_name} × {solver_name} ... ",
                    end="", flush=True,
                )
                res = run_one(scene_path, solver_name, args.timeout)
                row = {
                    "scene": scene_name,
                    "solver": solver_name,
                    "num_robots": meta["num_robots"],
                    "num_obstacles": meta["num_obstacles"],
                    "status": res.get("status", "crash"),
                    "elapsed_sec": res.get("elapsed_sec", ""),
                    "num_paths": res.get("num_paths", ""),
                    "max_waypoints": res.get("max_waypoints", ""),
                    "verify_ok": res.get("verify_ok", ""),
                    "notes": (res.get("notes") or "").replace("\n", " ")[:300],
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"{row['status']:>8}  "
                    f"{row['elapsed_sec']}s  "
                    f"paths={row['num_paths']} verify={row['verify_ok']}"
                )

    print(f"\nDone. Results → {args.out}")


if __name__ == "__main__":
    main()
