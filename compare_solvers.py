import argparse
import csv
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")

_SAMPLE_CAP = 15_000


def _scale(n_robots: int, base: int, cap: int = _SAMPLE_CAP) -> int:
    """Sample-budget scaling: super-linear in robot count (joint config
    space dimension is 2n) and capped to avoid blowing up runtime."""
    n = max(1, n_robots)
    return int(min(cap, base * (1 + (n - 1) ** 2)))


def _make_pprm(n_robots: int):
    from pprm_solver import pPRMSolver
    # num_samples is per-(cell, timestep), so it scales with cell-local n.
    ns = min(120, 30 + 10 * n_robots)
    params = f"num_samples={ns},k_nearest=8"
    return (
        pPRMSolver(
            num_samples=ns, k_nearest=8, max_cell_density=100,
        ),
        params,
    )


def _make_prm(n_robots: int):
    from discopygal.solvers.prm import PRM
    n_landmarks = _scale(n_robots, base=100)
    params = f"num_landmarks={n_landmarks},k_nn=15"
    return PRM(num_landmarks=n_landmarks, k_nn=15), params


def _make_rrt(n_robots: int):
    from discopygal.solvers.rrt import RRT
    n_landmarks = _scale(n_robots, base=200)
    params = f"num_landmarks={n_landmarks},eta=0.5"
    return RRT(num_landmarks=n_landmarks, eta=0.5), params


def _make_birrt(n_robots: int):
    from discopygal.solvers.rrt import BiRRT
    n_landmarks = _scale(n_robots, base=200)
    params = f"num_landmarks={n_landmarks},eta=0.5,n_join=5"
    return BiRRT(n_join=5, num_landmarks=n_landmarks, eta=0.5), params


def _make_drrt(n_robots: int):
    from discopygal.solvers.rrt import dRRT
    prm_l = _scale(n_robots, base=100)
    l = _scale(n_robots, base=100)
    params = f"num_landmarks={l},k_nn=15,prm_num_landmarks={prm_l}"
    return (
        dRRT(num_landmarks=l, k_nn=15, prm_num_landmarks=prm_l),
        params,
    )


def _make_drrt_star(n_robots: int):
    from discopygal.solvers.rrt import dRRT_star
    # num_expands dominates runtime; cap it tighter than _SAMPLE_CAP.
    expands = min(3000, _scale(n_robots, base=200, cap=10_000))
    prm_l = _scale(n_robots, base=100)
    params = (
        f"num_expands={expands},num_landmarks={prm_l},"
        f"k_nn=15,random_sample_counter=10"
    )
    return (
        dRRT_star(
            num_expands=expands, random_sample_counter=10,
            num_landmarks=prm_l, k_nn=15,
        ),
        params,
    )


def _make_staggered_grid(n_robots: int):
    from discopygal.solvers.staggered_grid import StaggeredGrid
    # Grid-based, not sample-count-driven; fix eps/delta and let the
    # wall-clock cap bound tensor-search cost.
    params = "eps=0.01,delta=0.1"
    return (
        StaggeredGrid(
            eps=0.01, delta=0.1, bounding_margin_width_factor=2,
        ),
        params,
    )


def _make_exact_single(n_robots: int):
    from discopygal.solvers.exact import ExactSingle
    # Single-robot only; raises on n>1 (recorded as "crash", intended).
    params = "eps=0.1"
    return ExactSingle(eps=0.1), params


SOLVERS: Dict[str, Callable[[int], Any]] = {
    "pPRM": _make_pprm,
    "PRM": _make_prm,
    "RRT": _make_rrt,
    "BiRRT": _make_birrt,
    "dRRT": _make_drrt,
    "dRRT_star": _make_drrt_star,
    "StaggeredGrid": _make_staggered_grid,
    "ExactSingle": _make_exact_single,
}

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
    """Resolve `selected` (basenames) or list scenes/ in benchmark order."""
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
    known = [available[n] for n in _DEFAULT_SCENE_ORDER if n in available]
    leftover = sorted(
        available[n] for n in available if n not in _DEFAULT_SCENE_ORDER
    )
    return known + leftover


def _child(
        scene_path: str, solver_name: str, out_path: str, n_robots: int,
) -> None:
    """Run one (scene, solver) trial in a child process and write the
    result dict to out_path as JSON. Catches every exception so the
    parent always sees a result file."""
    result: Dict[str, Any] = {}
    try:
        from discopygal.solvers_infra import Scene
        from discopygal.solvers_infra.verify_paths import verify_paths
        scene = Scene.from_file(scene_path)
        scene._source_path = scene_path
        factory = SOLVERS[solver_name]
        solver, params = factory(n_robots)
        result["params"] = params
        try:
            solver.disable_verbose()
        except Exception:
            pass
        # StaggeredGrid reads self._bounding_box in load_scene but never
        # populates it; fill it in when the solver opted into the knob.
        margin = getattr(solver, "bounding_margin_width_factor", -1)
        if (getattr(solver, "_bounding_box", None) is None
                and margin is not None and margin >= 0):
            solver.scene = scene
            try:
                solver._bounding_box = solver.calc_bounding_box()
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
        scene_path: str, solver_name: str, timeout: float, n_robots: int,
) -> Dict[str, Any]:
    """Spawn _child for one (scene, solver) trial with a wall-clock cap.

    :param scene_path: scene JSON path.
    :param solver_name: key into SOLVERS.
    :param timeout: seconds before the child is killed.
    :param n_robots: scene's robot count, used for sample scaling.
    :return: result dict with status / elapsed / paths / verify_ok / notes.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_child,
        args=(scene_path, solver_name, tmp.name, n_robots),
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


def scene_metadata(path: str) -> Dict[str, int]:
    """{'num_robots', 'num_obstacles'} from the scene JSON."""
    with open(path) as f:
        d = json.load(f)
    return {
        "num_robots": len(d.get("robots", [])),
        "num_obstacles": len(d.get("obstacles", [])),
    }


CSV_FIELDS = [
    "scene", "solver", "num_robots", "num_obstacles", "params",
    "status", "elapsed_sec", "num_paths", "max_waypoints",
    "verify_ok", "notes",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark pPRMSolver vs discopygal solvers on the scenes/ suite.",
    )
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
                    f"[{i:>3}/{total}] {scene_name} x {solver_name} ... ",
                    end="", flush=True,
                )
                res = run_one(
                    scene_path, solver_name, args.timeout,
                    meta["num_robots"],
                )
                row = {
                    "scene": scene_name,
                    "solver": solver_name,
                    "num_robots": meta["num_robots"],
                    "num_obstacles": meta["num_obstacles"],
                    "params": res.get("params", ""),
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
                    + (f"  [{row['params']}]" if row['params'] else "")
                )

    print(f"\nDone. Results -> {args.out}")


if __name__ == "__main__":
    main()
