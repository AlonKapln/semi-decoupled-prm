"""Partition the Minkowski-inflated free space with a regular grid
overlay, emitting one Partition per free face."""

import math
from typing import List, Tuple

from discopygal.bindings import (
    Aos2,
    Arr_overlay_function_traits,
    Arrangement_2,
    Curve_2,
    FT,
    Point_2,
    Pol2,
    Segment_2,
)
from discopygal.solvers_infra import Scene

from partition import Partition
from free_space_builder import FREE, construct_free_space


def _face_to_polygon(face) -> Pol2.Polygon_2:
    """Walk the outer CCB of a free face into a Polygon_2.

    CGAL coordinates live in an extended type (a0 + a1 * sqrt(gamma));
    projecting via a0() yields a rational subset of the true face.
    Safe (never claims blocked area as free).
    """
    poly = Pol2.Polygon_2()
    for halfedge in face.outer_ccb():
        src = halfedge.source().point()
        poly.push_back(Point_2(src.x().a0(), src.y().a0()))
    if poly.is_clockwise_oriented():
        poly.reverse_orientation()
    return poly


def _refine_with_grid(
        arr: Arrangement_2,
        robot_radius: float,
        max_cell_density: int,
) -> Arrangement_2:
    """Overlay axis-aligned grid cuts so every resulting cell has area
    at most max_cell_density * pi * (2r)^2. Spacing is floored at 4r so
    r-margin sampling stays feasible."""
    x_vals = []
    y_vals = []
    for v in arr.vertices():
        x_vals.append(v.point().x().a0())
        y_vals.append(v.point().y().a0())

    if not x_vals:
        return arr

    min_x, max_x = min(x_vals), max(x_vals)
    y_sorted = sorted(set(y_vals), key=lambda ft: ft.to_double())
    min_y, max_y = y_sorted[0], y_sorted[-1]

    grid_spacing = max(
        math.sqrt(max_cell_density * math.pi * (2.0 * robot_radius) ** 2),
        4.0 * robot_radius,
    )

    walls = []

    dy = max_y.to_double() - min_y.to_double()
    n_h = int(dy / grid_spacing) + 1
    for i in range(1, n_h):
        yv = min_y.to_double() + i * grid_spacing
        if yv < max_y.to_double():
            walls.append(Curve_2(Segment_2(
                Point_2(min_x, FT(yv)), Point_2(max_x, FT(yv)),
            )))

    dx = max_x.to_double() - min_x.to_double()
    n_v = int(dx / grid_spacing) + 1
    for i in range(1, n_v):
        xv = min_x.to_double() + i * grid_spacing
        if xv < max_x.to_double():
            walls.append(Curve_2(Segment_2(
                Point_2(FT(xv), min_y), Point_2(FT(xv), max_y),
            )))

    if not walls:
        return arr

    walls_arr = Arrangement_2()
    Aos2.insert(walls_arr, walls)
    for face in walls_arr.faces():
        face.set_data(FREE)

    traits = Arr_overlay_function_traits(lambda x, y: x + y)
    return Aos2.overlay(arr, walls_arr, traits)


def partition_free_space_grid(
        scene: Scene,
        robot_radius: float,
        eps: float = 1e-4,
        max_cell_density: int = 100,
) -> Tuple[List[Partition], Arrangement_2]:
    """Free-space grid partition: Minkowski-inflated arrangement overlaid
    with an axis-aligned grid, one Partition per free face. Cells are
    not guaranteed convex (arc notches along inflated obstacles); the
    downstream pipeline handles this with point-in-polygon + the
    scene-level collision checker."""
    arrangement = construct_free_space(scene, robot_radius=robot_radius, eps=eps)
    arrangement = _refine_with_grid(arrangement, robot_radius, max_cell_density)

    partitions: List[Partition] = []
    for face in arrangement.faces():
        if face.is_unbounded() or face.data() != FREE:
            continue
        poly = _face_to_polygon(face)
        if poly.size() < 3:
            continue
        partitions.append(Partition(polygon=poly, robot_radius=robot_radius))
    return partitions, arrangement
