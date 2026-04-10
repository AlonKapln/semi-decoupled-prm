from typing import List, Tuple

from discopygal.bindings import (
    Aos2,
    Arr_overlay_function_traits,
    Arrangement_2,
    Curve_2,
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


def partition_free_space_vertical(
        scene: Scene,
        robot_radius: float,
        eps: float = 1e-4,
) -> Tuple[List[Partition], Arrangement_2]:
    """Build the disc-robot free space and partition it into trapezoidal cells.

    This implements steps 1+2 of ``plan.tex``:

    1. ``construct_free_space`` builds an arrangement of inflated obstacles
       clipped to the bounding box (faces tagged ``FREE`` / ``BLOCKED``).
    2. ``_vertical_decompose`` adds vertical walls at every vertex, splitting
       every free face into convex trapezoids/triangles.
    3. Each free face is then converted to a ``Pol2.Polygon_2`` and wrapped
       in a ``Partition`` whose ``density`` is computed from area and
       ``robot_radius``.

    The returned arrangement is preserved so downstream stages (the
    high-level graph builder) can walk halfedges to find shared boundaries
    between adjacent cells.
    """
    arr = construct_free_space(scene, robot_radius=robot_radius, eps=eps)
    arr = _vertical_decompose(arr)

    partitions: List[Partition] = []
    for face in arr.faces():
        if face.is_unbounded() or face.data() != FREE:
            continue
        poly = _face_to_polygon(face)
        if poly.size() < 3:
            continue
        partitions.append(Partition(polygon=poly, robot_radius=robot_radius))
    return partitions, arr
