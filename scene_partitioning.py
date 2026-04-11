import math
from typing import List, Tuple

from discopygal.bindings import (
    Aos2,
    Arr_overlay_function_traits,
    Arrangement_2,
    Curve_2,
    FT,
    Halfedge,
    Ker,
    Point_2,
    Pol2,
    Segment_2,
    Vertex,
)
from discopygal.solvers_infra import Scene

from partition import Partition
from free_space_builder import FREE, construct_free_space


def _to_ker_point_2(point) -> "Ker.Point_2":
    """Convert a CGAL extended ``TPoint`` (used for arc endpoints) to a plain
    ``Ker.Point_2`` by taking only the rational part. Mirrors
    ``ExactSingle.to_ker_point_2``."""
    return Ker.Point_2(point.x().a0(), point.y().a0())


def _vertical_decompose(arr: Arrangement_2) -> Aos2:
    """Add vertical walls to ``arr`` so every face is a trapezoid.

    Lifted from ``ExactSingle.vertical_decomposition``. ``Aos2.decompose``
    returns, for each vertex of the arrangement, the topmost and bottommost
    objects (vertices or halfedges) directly above/below it. We turn each
    such (vertex, object) pair into a vertical line segment and overlay the
    resulting walls back onto the arrangement.

    The face ``data()`` tags from the input are preserved by the overlay
    (their sum is unchanged because the wall arrangement uses ``data() == 0``).
    """
    decomposition = Aos2.decompose(arr)
    walls = []
    for vertex, neighbours in decomposition:
        v_point = _to_ker_point_2(vertex.point())
        for obj in neighbours:
            if isinstance(obj, Vertex):
                other = _to_ker_point_2(obj.point())
                walls.append(Curve_2(Segment_2(v_point, other)))
            elif isinstance(obj, Halfedge):
                line = Ker.Line_2(
                    _to_ker_point_2(obj.source().point()),
                    _to_ker_point_2(obj.target().point()),
                )
                y_at_x = line.y_at_x(v_point.x())
                walls.append(
                    Curve_2(Segment_2(v_point, Point_2(v_point.x(), y_at_x)))
                )

    walls_arr = Arrangement_2()
    Aos2.insert(walls_arr, walls)
    for face in walls_arr.faces():
        face.set_data(FREE)

    traits = Arr_overlay_function_traits(lambda x, y: x + y)
    return Aos2.overlay(arr, walls_arr, traits)


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

    After vertical decomposition, large open faces may still have very high
    density. This function adds a **regular grid** of horizontal and vertical
    cuts whose spacing is chosen so that each resulting cell has area at most
    ``max_cell_density * pi * (2r)^2`` — i.e. density <= max_cell_density.

    The grid spacing is also floored at ``4 * robot_radius`` so that every
    cell is wide enough for boundary-inset sampling (points kept at least
    ``r`` from all edges).

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


def partition_free_space_vertical(
        scene: Scene,
        robot_radius: float,
        eps: float = 1e-4,
        max_cell_density: int = 4,
) -> Tuple[List[Partition], Arrangement_2]:
    """Build the disc-robot free space and partition it into convex cells.

    This implements steps 1+2 of ``plan.tex``:

    1. ``construct_free_space`` builds an arrangement of inflated obstacles
       clipped to the bounding box (faces tagged ``FREE`` / ``BLOCKED``).
    2. ``_vertical_decompose`` adds vertical walls at every vertex, splitting
       every free face into convex trapezoids/triangles.
    3. ``_refine_with_grid`` adds horizontal walls at every vertex plus a
       regular grid of cuts so that no cell exceeds ``max_cell_density``.
    4. Each free face is converted to a ``Pol2.Polygon_2`` and wrapped in a
       ``Partition`` whose ``density`` is computed from area and
       ``robot_radius``.

    Parameters
    ----------
    max_cell_density :
        Target upper bound for per-cell density.  Grid cuts are spaced so
        that each cell's area yields density <= this value.  Lower values
        give smaller cells (better for joint PRM) but more cells (slower
        high-level graph and MCF).  Default 8.
    """
    arr = construct_free_space(scene, robot_radius=robot_radius, eps=eps)
    arr = _vertical_decompose(arr)
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


def partition_free_space_grid(
        scene: Scene,
        robot_radius: float,
        eps: float = 1e-4,
        max_cell_density: int = 4,
) -> Tuple[List[Partition], Arrangement_2]:
    """Partition free space with a *pure grid* — no vertical decomposition.

    Alternative to ``partition_free_space_vertical`` for scenes where
    vertical decomposition produces narrow slivers (cells narrower than
    ``2r``) that the ad-hoc joint PRM cannot separate safely. By skipping
    ``_vertical_decompose`` entirely and relying solely on the regular grid
    in ``_refine_with_grid``, cells are coarser and stay comfortably wide
    as long as ``max_cell_density`` is chosen so the grid spacing is at
    least ``2r`` on each side (the ``_refine_with_grid`` floor of ``4r``
    already guarantees this).

    Trade-offs
    ----------
    - Cells are **not guaranteed convex**. The inflated-obstacle boundaries
      carve curved notches into cells that touch them. The downstream code
      (``build_adhoc_roadmap``, ``_inset_toward_centroid``) already handles
      non-convex partitions via closed point-in-polygon tests.
    - For a given ``max_cell_density`` the grid version typically yields
      *fewer*, bigger cells than the vertical+grid version, so the HLG and
      MCF are smaller. Trade: less precise capacity control inside large
      open regions.
    - Use this mode on scenes with axis-aligned obstacles and narrow
      corridors (warehouses, office layouts). Prefer the vertical mode for
      scenes with diagonal obstacles where vertical cuts align with
      natural cell boundaries.

    Parameters
    ----------
    max_cell_density :
        Passed through to ``_refine_with_grid``. Same meaning as in
        ``partition_free_space_vertical``.
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
