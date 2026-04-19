"""Build the configuration-space free region for disc robots.

Step 1 of ``plan.tex``.

Geometry
--------
For a disc robot of radius ``r`` moving among polygonal obstacles ``O_i``,
the *configuration-space obstacle* of each ``O_i`` is its Minkowski sum with a
disc of radius ``r``::

    C(O_i) = O_i ⊕ D(r) = { o + d : o ∈ O_i, d ∈ D(r) }

A robot center can occupy any point that lies outside every ``C(O_i)`` and
inside the workspace boundary. The union of all such points is the
*configuration-space free region* ``C_free``.

We do not store ``C_free`` as one polygon. Instead we build a CGAL
``Arrangement_2`` whose edges are the boundaries of all ``C(O_i)`` plus the
bounding-box edges. The arrangement partitions the plane into faces; we tag
each face's ``data()`` as ``FREE`` or ``BLOCKED``. This arrangement is the
exact substrate that the trapezoidal decomposition (step 2) consumes.

Pipeline (per obstacle)
-----------------------
1. ``Ms2.approximated_offset_2(obstacle.poly, r, eps)`` returns a
   ``Polygon_with_holes_2`` whose outer boundary is a polyline-with-arcs
   approximation of ``∂C(O_i)`` to within ``eps``.
2. The boundary curves are inserted into a fresh ``Arrangement_2``.
3. The face just inside the outer boundary is the inflated obstacle and is
   tagged ``BLOCKED``; everything else is ``FREE``.

All per-obstacle arrangements are then overlaid (``Aos2.overlay``), and a
final overlay with the bounding-box arrangement clips the result to the
workspace.

Why an arrangement instead of a polygon-set
-------------------------------------------
``discopygal.bindings.Bso2`` only exposes Boolean operations on the
``Aos2.Gps_traits_2`` general polygon family, while ``Ms2`` returns
``Pol2.Polygon_with_holes_2``. The two type families are not directly
compatible. Working at the arrangement level sidesteps the conversion and is
exactly what ``discopygal.solvers.exact.exact_single`` does — we lift its
``construct_cspace`` so we inherit its tested CGAL plumbing.
"""

from typing import Optional, Tuple

from discopygal.bindings import (
    Arrangement_2,
    Aos2,
    Arr_overlay_function_traits,
    Curve_2,
    FT,
    Ms2,
    Point_2,
)
from discopygal.geometry_utils import bounding_boxes, conversions
from discopygal.solvers_infra import Scene, ObstaclePolygon

# Face data values
FREE = 0
BLOCKED = 1


def construct_free_space(
    scene: Scene,
    robot_radius: float,
    eps: float = 1e-4,
    bounding_box: Optional[Tuple[FT, FT, FT, FT]] = None,
) -> Arrangement_2:
    """Return a CGAL arrangement representing the disc-robot free space.

    Each face's ``data()`` is ``FREE`` (0) for traversable area or ``BLOCKED`` (1)
    for inflated obstacles / outside the bounding box.

    Args:
        scene: discopygal scene with disc robots and polygonal obstacles.
        robot_radius: radius used to inflate obstacles.
        eps: precision passed to ``Ms2.approximated_offset_2``.
        bounding_box: optional ``(min_x, max_x, min_y, max_y)`` in CGAL ``FT``;
            defaults to ``calc_scene_bounding_box(scene)``.
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
        # The face just inside the outer boundary is the inflated obstacle.
        invalid_face = next(next(ubf.inner_ccbs())).twin().face()
        invalid_face.set_data(BLOCKED)
        for ccb in invalid_face.inner_ccbs():
            valid_face = next(ccb).twin().face()
            valid_face.set_data(FREE)

        arrangements.append(arr)

    # Overlay all per-obstacle arrangements
    initial = Arrangement_2()
    initial.unbounded_face().set_data(FREE)
    arrangements.insert(0, initial)
    arr = initial
    for i in range(len(arrangements) - 1):
        arr = Aos2.overlay(arrangements[i], arrangements[i + 1], traits)
        arrangements[i + 1] = arr

    # Bounding-box arrangement: inside == FREE, unbounded == BLOCKED
    if bounding_box is None:
        bounding_box = bounding_boxes.calc_scene_bounding_box(scene)
    min_x, max_x, min_y, max_y = bounding_box

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
