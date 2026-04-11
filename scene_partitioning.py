"""Partition the Minkowski-inflated free space into small convex cells.

Currently only a pure grid decomposition is supported via
``partition_free_space_grid``. An earlier trapezoidal-decomposition pass
(``_vertical_decompose`` + ``partition_free_space_vertical``) was removed:
it produced narrow slivers along inflated-obstacle arcs that the joint
PRM could not reliably separate, and it offered no solve-success benefit
over the grid decomposition once ``max_cell_density`` was tuned.
"""

import math
from typing import List, Tuple

from discopygal.bindings import (
    Aos2,
    Arr_overlay_function_traits,
    Arrangement_2,
    Curve_2,
    FT,
    Ker,
    Point_2,
    Pol2,
    Segment_2,
)
from discopygal.solvers_infra import Scene

from partition import Partition
from free_space_builder import FREE, construct_free_space


def _to_ker_point_2(point) -> "Ker.Point_2":
    """Convert a CGAL extended ``TPoint`` (used for arc endpoints) to a plain
    ``Ker.Point_2`` by taking only the rational part. Mirrors
    ``ExactSingle.to_ker_point_2``."""
    return Ker.Point_2(point.x().a0(), point.y().a0())


def _face_to_polygon(face) -> Pol2.Polygon_2:
    """Walk the outer CCB of a free face and emit a CGAL ``Polygon_2``.

    The vertices of an arrangement built from ``approximated_offset_2`` live
    in CGAL's *extended* number type (each coordinate is ``a0 + a1·sqrt(γ)``)
    so that circular arc intersections are exact. The trapezoidal cells we
    care about have only straight edges, but their endpoints can still sit on
    arc intersection points. We project to the rational part via ``a0()``;
    the resulting cell is a strict subset of the true free face which is
    safe (it never claims blocked area as free).
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
    """Add a regular grid of cuts to limit cell density.

    Adds horizontal and vertical cut lines whose spacing is chosen so that
    each resulting cell has area at most
    ``max_cell_density * pi * (2r)^2`` — i.e. density <= max_cell_density.
    Spacing is floored at ``4 * robot_radius`` so every cell is wide enough
    for boundary-inset sampling (points kept at least ``r`` from all edges).

    All cuts span the full bounding box. The overlay preserves face data
    tags (free/blocked) via additive functor.
    """
    # Collect bounding box
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

    # Grid spacing: target area density, but at least 4r so cells are
    # wide enough for r-inset sampling.
    grid_spacing = max(
        math.sqrt(max_cell_density * math.pi * (2.0 * robot_radius) ** 2),
        4.0 * robot_radius,
    )

    walls = []

    # Horizontal grid lines
    dy = max_y.to_double() - min_y.to_double()
    n_h = int(dy / grid_spacing) + 1
    for i in range(1, n_h):
        yv = min_y.to_double() + i * grid_spacing
        if yv < max_y.to_double():
            walls.append(Curve_2(Segment_2(
                Point_2(min_x, FT(yv)), Point_2(max_x, FT(yv)),
            )))

    # Vertical grid lines
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
    """Partition free space with a regular grid.

    1. ``construct_free_space`` builds an arrangement of inflated obstacles
       clipped to the bounding box (faces tagged ``FREE`` / ``BLOCKED``).
    2. ``_refine_with_grid`` overlays a regular grid of horizontal and
       vertical cuts so that each resulting cell has area bounded by
       ``max_cell_density`` disc-packing units. Grid spacing is floored at
       ``4 * robot_radius`` so every cell is wide enough for boundary-inset
       sampling.
    3. Each free face is converted to a ``Pol2.Polygon_2`` and wrapped in a
       ``Partition`` whose ``density`` is computed from area and radius.

    Cells are **not guaranteed convex**: inflated-obstacle arcs carve curved
    notches into cells that touch them. The downstream code
    (``build_adhoc_roadmap``, ``high_level_graph``) handles non-convex
    partitions via closed point-in-polygon tests plus a scene-level
    collision checker for exact-arc validity.

    Parameters
    ----------
    max_cell_density :
        Upper bound for per-cell density. Lower values split large open
        regions into smaller cells (better joint-PRM performance, slower
        HLG/MCF). Higher values leave cells as large as the free-space
        topology allows.
    """
    arr = construct_free_space(scene, robot_radius=robot_radius, eps=eps)
    arr = _refine_with_grid(arr, robot_radius, max_cell_density)

    partitions: List[Partition] = []
    for face in arr.faces():
        if face.is_unbounded() or face.data() != FREE:
            continue
        poly = _face_to_polygon(face)
        if poly.size() < 3:
            continue
        partitions.append(Partition(polygon=poly, robot_radius=robot_radius))
    return partitions, arr
