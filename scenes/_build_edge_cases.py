"""Generate edge-case scenes into scenes/.

Run once: ``python3 scenes/_build_edge_cases.py``. Produces JSON files
consumable by ``Scene.from_file`` and by the benchmark script. Some
scenes are intentionally hard or unsolvable for the staged solver — that
is the point: the benchmark highlights where each planner breaks.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _robot(start, end, radius=0.3, color="blue", name=""):
    return {
        "__class__": "RobotDisc",
        "radius": radius,
        "start": list(start),
        "end": list(end),
        "data": {"color": color, "name": name, "value": "", "details": ""},
    }


def _obstacle(vertices):
    return {
        "__class__": "ObstaclePolygon",
        "poly": [list(v) for v in vertices],
        "data": {},
    }


def _rect(x0, y0, x1, y1):
    """CCW rectangle with corners (x0, y0) to (x1, y1)."""
    return _obstacle([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _scene(robots, obstacles, note=""):
    return {
        "__class__": "Scene",
        "obstacles": obstacles,
        "robots": robots,
        "metadata": {"note": note},
    }


def _dump(name, scene):
    path = os.path.join(HERE, f"{name}.json")
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def swap_2():
    """Two discs, no obstacles, head-on swap — trivial golden path."""
    r = 0.3
    return _scene(
        robots=[
            _robot((-4, 0), (4, 0), r, "red", "A"),
            _robot((4, 0), (-4, 0), r, "blue", "B"),
        ],
        obstacles=[_rect(-10, -5, -9.99, 5)],  # dummy wall to anchor bbox
        note="2-robot head-on swap in open space",
    )


def rotate_6():
    """Six robots on a circle, each rotates one position (60 deg).

    Classic dense-rotation test: every robot's goal is another robot's
    start, so the group must collectively rotate.
    """
    r = 0.4
    R = 3.0
    n = 6
    robots = []
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    for i in range(n):
        a = 2 * math.pi * i / n
        b = 2 * math.pi * ((i + 1) % n) / n
        robots.append(_robot(
            (R * math.cos(a), R * math.sin(a)),
            (R * math.cos(b), R * math.sin(b)),
            r, colors[i], f"R{i}",
        ))
    return _scene(
        robots=robots,
        obstacles=[_rect(-7, -7, -6.99, 7)],
        note="6-robot dense rotation on a circle",
    )


def narrow_corridor():
    """Two robots head-on in a corridor ~4r wide — barely passable."""
    r = 0.3
    gap = 4.0 * r  # 1.2 — just above 2r so they can shuffle past
    obstacles = [
        _rect(-10, gap / 2, 10, 3),        # top wall
        _rect(-10, -3, 10, -gap / 2),      # bottom wall
    ]
    return _scene(
        robots=[
            _robot((-8, 0), (8, 0), r, "red", "A"),
            _robot((8, 0), (-8, 0), r, "blue", "B"),
        ],
        obstacles=obstacles,
        note="2-robot head-on in 4r-wide corridor",
    )


def bottleneck_funnel():
    """Four robots left→right through a 1-gap funnel.

    The only passage is a single gap the width of ~3r centred on y=0.
    Serialization is required.
    """
    r = 0.25
    half_gap = 1.5 * r  # 0.375 — ~3r wide
    obstacles = [
        _rect(-0.5, half_gap, 0.5, 5),      # top jaw
        _rect(-0.5, -5, 0.5, -half_gap),    # bottom jaw
        _rect(-10, -5.01, 10, -5),          # floor (context)
        _rect(-10, 5, 10, 5.01),            # ceiling
    ]
    ys = [-3.0, -1.0, 1.0, 3.0]
    robots = []
    colors = ["red", "blue", "green", "orange"]
    for i, y in enumerate(ys):
        robots.append(_robot(
            (-5.0, y), (5.0, y), r, colors[i], f"R{i}",
        ))
    return _scene(
        robots=robots,
        obstacles=obstacles,
        note="4 robots through a single narrow gap",
    )


def L_corridor_single():
    """1 robot through an L-shaped corridor.

    Should be trivial for PRM/RRT; useful as a single-robot baseline.
    """
    r = 0.2
    obstacles = [
        _rect(-5, -5, -2, 2),       # lower-left block
        _rect(-2, 2, 3, 5),         # upper-mid block
        _rect(3, -5, 5, 0),         # lower-right block
    ]
    return _scene(
        robots=[_robot((-4.5, -4.5), (4.5, -4.5), r, "red", "R")],
        obstacles=obstacles,
        note="single robot, L-shaped corridor",
    )


def maze_single():
    """1 robot in a small obstacle maze."""
    r = 0.2
    walls = [
        _rect(-5, -5, 5, -4.8),         # floor
        _rect(-5, 4.8, 5, 5),           # ceiling
        _rect(-5, -5, -4.8, 5),         # left
        _rect(4.8, -5, 5, 5),           # right
        _rect(-3, -3, -2, 3),           # interior wall 1
        _rect(0, -4, 1, 2),             # interior wall 2
        _rect(2.5, -2, 3.5, 4),         # interior wall 3
    ]
    return _scene(
        robots=[_robot((-4, -4), (4, 4), r, "red", "R")],
        obstacles=walls,
        note="single robot, boxed maze with 3 interior walls",
    )


def crossing_4():
    """4 robots at compass points swap with their opposite.

    Classic deadlock-prone 4-way crossing with no obstacles.
    """
    r = 0.35
    return _scene(
        robots=[
            _robot((-4, 0), (4, 0), r, "red", "E"),
            _robot((4, 0), (-4, 0), r, "blue", "W"),
            _robot((0, -4), (0, 4), r, "green", "N"),
            _robot((0, 4), (0, -4), r, "orange", "S"),
        ],
        obstacles=[_rect(-6, -5.01, 6, -5)],
        note="4-way compass swap, no obstacles",
    )


def open_16():
    """16 robots in an open 20x20 area doing antipodal swaps.

    Heavy multi-robot stress with no obstacles; exercises the router.
    """
    r = 0.25
    n = 16
    robots = []
    for i in range(n):
        a = 2 * math.pi * i / n
        R_outer = 8.0
        sx, sy = R_outer * math.cos(a), R_outer * math.sin(a)
        ex, ey = -sx, -sy
        robots.append(_robot((sx, sy), (ex, ey), r, "blue", f"R{i}"))
    return _scene(
        robots=robots,
        obstacles=[_rect(-10, -10.01, 10, -10)],
        note="16 robots on a circle doing antipodal swaps",
    )


def spiral_4():
    """4 robots in a spiral pattern around a central obstacle."""
    r = 0.3
    obstacles = [
        _rect(-1, -1, 1, 1),           # central square
        _rect(-3.5, -3.5, -2.5, 3),    # left column
        _rect(2.5, -3, 3.5, 3.5),      # right column
    ]
    return _scene(
        robots=[
            _robot((-5, -5), (5, 5), r, "red", "A"),
            _robot((5, -5), (-5, 5), r, "blue", "B"),
            _robot((-5, 5), (5, -5), r, "green", "C"),
            _robot((5, 5), (-5, -5), r, "orange", "D"),
        ],
        obstacles=obstacles,
        note="4 robots, central + 2 side obstacles",
    )


def single_robot_empty():
    """Single robot, no obstacles. Should be trivial for everyone."""
    return _scene(
        robots=[_robot((-4, -4), (4, 4), 0.3, "red", "R")],
        obstacles=[_rect(-5, -5.01, 5, -5)],
        note="single robot, straight-line path, no obstacles",
    )


def unsolvable_trap():
    """Two robots, the second's goal is inside a sealed box.

    Intentionally unsolvable — every solver should either return no
    path or time out. Useful to confirm solvers fail cleanly.
    """
    r = 0.3
    obstacles = [
        # Sealed box around (3, 3)
        _rect(2, 2, 4, 2.2),
        _rect(2, 3.8, 4, 4),
        _rect(2, 2, 2.2, 4),
        _rect(3.8, 2, 4, 4),
    ]
    return _scene(
        robots=[
            _robot((-4, -4), (4, -4), r, "red", "A"),
            _robot((-4, 0), (3, 3), r, "blue", "B"),  # B's goal is inside sealed box
        ],
        obstacles=obstacles,
        note="unsolvable: second robot's goal is inside a sealed box",
    )


SCENES = [
    ("single_robot_empty", single_robot_empty),
    ("L_corridor_single", L_corridor_single),
    ("maze_single", maze_single),
    ("swap_2", swap_2),
    ("narrow_corridor", narrow_corridor),
    ("crossing_4", crossing_4),
    ("bottleneck_funnel", bottleneck_funnel),
    ("spiral_4", spiral_4),
    ("rotate_6", rotate_6),
    ("open_16", open_16),
    ("unsolvable_trap", unsolvable_trap),
]


if __name__ == "__main__":
    for name, fn in SCENES:
        _dump(name, fn())
    print(f"\nBuilt {len(SCENES)} scenes.")
