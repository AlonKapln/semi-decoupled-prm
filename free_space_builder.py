from discopygal.bindings import (
    Arrangement_2,
    Aos2,
    Arr_overlay_function_traits,
    Curve_2,
    Ms2,
    Point_2,
)
from discopygal.geometry_utils import bounding_boxes, conversions
from discopygal.solvers_infra import Scene, ObstaclePolygon

FREE = 0
BLOCKED = 1


def _tree_overlay(arrs, traits) -> Arrangement_2:
    """Balanced pairwise merge of arrangements: log n depth vs. the
    quadratic left-fold accumulator."""
    assert arrs, "_tree_overlay needs at least one arrangement"
    while len(arrs) > 1:
        merged = [
            Aos2.overlay(arrs[i], arrs[i + 1], traits)
            for i in range(0, len(arrs) - 1, 2)
        ]
        if len(arrs) % 2 == 1:
            merged.append(arrs[-1])
        arrs = merged
    return arrs[0]


def construct_free_space(
    scene: Scene,
    robot_radius: float,
    eps: float = 1e-4,
) -> Arrangement_2:
    """Build the Minkowski-inflated free-space arrangement.

    Each obstacle is inflated by the robot radius (CGAL's approximated
    offset), the per-obstacle arrangements are merged, and the result is
    overlaid with the scene bounding box. Each face's data is FREE (0)
    for traversable area or BLOCKED (1) for inflated obstacles / outside
    the box.

    :param scene: discopygal scene.
    :param robot_radius: disc radius r.
    :param eps: tolerance passed to approximated_offset_2.
    :return: arrangement with FREE/BLOCKED face data.
    """
    traits = Arr_overlay_function_traits(lambda x, y: x + y)

    arrangements = []
    for obstacle in scene.obstacles:
        if not isinstance(obstacle, ObstaclePolygon):
            continue
        arr = Arrangement_2()
        ms = Ms2.approximated_offset_2(obstacle.poly, robot_radius, eps)
        Aos2.insert(arr, conversions.to_list(ms.outer_boundary().curves()))
        for hole in ms.holes():
            Aos2.insert(arr, conversions.to_list(hole.curves()))

        ubf = arr.unbounded_face()
        ubf.set_data(FREE)
        # Face just inside the outer boundary is the inflated obstacle.
        invalid_face = next(next(ubf.inner_ccbs())).twin().face()
        invalid_face.set_data(BLOCKED)
        for ccb in invalid_face.inner_ccbs():
            valid_face = next(ccb).twin().face()
            valid_face.set_data(FREE)

        arrangements.append(arr)

    if not arrangements:
        # No obstacles: seed a FREE-everywhere arrangement so the
        # bounding-box overlay has something to merge against.
        empty = Arrangement_2()
        empty.unbounded_face().set_data(FREE)
        arrangements.append(empty)
    arr = _tree_overlay(arrangements, traits)

    min_x, max_x, min_y, max_y = bounding_boxes.calc_scene_bounding_box(scene)

    bounding_box_arr = Arrangement_2()
    Aos2.insert(
        bounding_box_arr,
        [
            Curve_2(Point_2(min_x, min_y), Point_2(max_x, min_y)),
            Curve_2(Point_2(max_x, min_y), Point_2(max_x, max_y)),
            Curve_2(Point_2(max_x, max_y), Point_2(min_x, max_y)),
            Curve_2(Point_2(min_x, max_y), Point_2(min_x, min_y)),
        ],
    )
    for face in bounding_box_arr.faces():
        face.set_data(BLOCKED if face.is_unbounded() else FREE)

    return Aos2.overlay(arr, bounding_box_arr, traits)
