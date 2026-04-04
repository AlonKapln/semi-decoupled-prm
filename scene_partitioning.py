import math
from itertools import combinations
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from sklearn.svm import SVC

from discopygal.bindings import Point_2, FT, Pol2
from discopygal.solvers_infra import Scene, ObstaclePolygon

from Partition import Partition


def _compute_bounding_box(scene: Scene, padding: float = 0.0):
    """Compute axis-aligned bounding box of the scene (robots + obstacles), with optional padding."""
    xs, ys = [], []

    for robot in scene.robots:
        for pt in (robot.start, robot.end):
            xs.append(pt.x().to_double())
            ys.append(pt.y().to_double())

    for obstacle in scene.obstacles:
        if isinstance(obstacle, ObstaclePolygon):
            for pt in obstacle.poly.vertices():
                xs.append(pt.x().to_double())
                ys.append(pt.y().to_double())

    return float(min(xs) - padding), float(max(xs) + padding), float(min(ys) - padding), float(max(ys) + padding)


def partition_scene(scene: Scene, cell_size: float = 1.0) -> List[Partition]:
    """
    Partition the scene into equal-sized square cells.

    Args:
        scene: A discopygal Scene with robots and obstacles.
        cell_size: Side length of each square cell.

    Returns:
        List of Partition objects covering the scene bounding box.
    """
    max_radius = max(robot.radius for robot in scene.robots)
    x_min, x_max, y_min, y_max = _compute_bounding_box(scene, padding=max_radius)

    cols = math.ceil((x_max - x_min) / cell_size)
    rows = math.ceil((y_max - y_min) / cell_size)

    partitions: List[Partition] = []
    for row in range(rows):
        for col in range(cols):
            x0 = x_min + col * cell_size
            y0 = y_min + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            poly = Pol2.Polygon_2()
            poly.push_back(Point_2(FT(x0), FT(y0)))
            poly.push_back(Point_2(FT(x1), FT(y0)))
            poly.push_back(Point_2(FT(x1), FT(y1)))
            poly.push_back(Point_2(FT(x0), FT(y1)))

            partitions.append(Partition(polygon=poly))

    return partitions


def _split_polygon_by_line(poly: ShapelyPolygon, w: np.ndarray, b: float,
                           big: float = 1e6) -> List[ShapelyPolygon]:
    """Split a shapely polygon by the line w·x + b = 0 into up to two sub-polygons.

    Constructs a large half-plane box on each side of the line and intersects
    with the polygon.
    """
    # Normal direction and perpendicular
    nx, ny = w[0], w[1]
    norm = math.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    # Perpendicular (tangent along the line)
    tx, ty = -ny, nx

    # Build a large quad for the positive half-plane (w·x + b >= 0)
    # Shift the line's base point: a point on the line is found by solving w·p + b = 0
    # Use the point closest to origin: p0 = -b * w / ||w||^2
    w_norm_sq = w[0] ** 2 + w[1] ** 2
    p0 = -b * w / w_norm_sq  # point on the line

    # Four corners of the positive half-plane box
    pos_quad = ShapelyPolygon([
        (p0[0] + big * tx + big * nx, p0[1] + big * ty + big * ny),
        (p0[0] - big * tx + big * nx, p0[1] - big * ty + big * ny),
        (p0[0] - big * tx, p0[1] - big * ty),
        (p0[0] + big * tx, p0[1] + big * ty),
    ])
    neg_quad = ShapelyPolygon([
        (p0[0] + big * tx - big * nx, p0[1] + big * ty - big * ny),
        (p0[0] - big * tx - big * nx, p0[1] - big * ty - big * ny),
        (p0[0] - big * tx, p0[1] - big * ty),
        (p0[0] + big * tx, p0[1] + big * ty),
    ])

    results = []
    for half_plane in (pos_quad, neg_quad):
        piece = poly.intersection(half_plane)
        if piece.is_empty:
            continue
        # intersection can produce MultiPolygon or GeometryCollection
        if piece.geom_type == 'Polygon' and piece.area > 1e-10:
            results.append(piece)
        elif piece.geom_type in ('MultiPolygon', 'GeometryCollection'):
            for geom in piece.geoms:
                if geom.geom_type == 'Polygon' and geom.area > 1e-10:
                    results.append(geom)

    return results


def _shapely_to_cgal_polygon(spoly: ShapelyPolygon) -> Pol2.Polygon_2:
    """Convert a shapely Polygon to a CGAL Polygon_2 (counter-clockwise)."""
    coords = list(spoly.exterior.coords)[:-1]  # drop closing duplicate vertex
    # Ensure counter-clockwise orientation (shapely uses CCW by default for valid polygons)
    poly = Pol2.Polygon_2()
    for x, y in coords:
        poly.push_back(Point_2(FT(x), FT(y)))
    if poly.is_clockwise_oriented():
        poly.reverse_orientation()
    return poly


def _extract_svm_lines(scene: Scene, C: float) -> List[Tuple[np.ndarray, float]]:
    """Train one-vs-one linear SVMs for each robot pair, return list of (w, b) lines."""
    n = len(scene.robots)
    points = []
    labels = []
    for i, robot in enumerate(scene.robots):
        points.append([robot.start.x().to_double(), robot.start.y().to_double()])
        points.append([robot.end.x().to_double(), robot.end.y().to_double()])
        labels.extend([i, i])

    points = np.array(points)
    labels = np.array(labels)

    lines = []
    seen_normals = []

    for i, j in combinations(range(n), 2):
        mask = (labels == i) | (labels == j)
        X = points[mask]
        y = labels[mask]
        # Relabel to +1/-1
        y_bin = np.where(y == i, 1, -1)

        # Skip if all points are identical (no separation possible)
        if np.allclose(X[y_bin == 1], X[y_bin == -1]):
            continue

        svm = SVC(kernel='linear', C=C)
        svm.fit(X, y_bin)
        w = svm.coef_[0]
        b = svm.intercept_[0]

        # Normalize w to unit length for deduplication
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w_unit = w / w_norm
        b_unit = b / w_norm

        # Skip near-duplicate lines (same direction and offset)
        is_dup = False
        for w_prev, b_prev in seen_normals:
            # Check if lines are essentially the same (parallel + same offset)
            dot = abs(np.dot(w_unit, w_prev))
            if dot > 0.999:
                # Same direction — check offset
                sign = np.sign(np.dot(w_unit, w_prev))
                if abs(b_unit - sign * b_prev) < 0.01:
                    is_dup = True
                    break
        if is_dup:
            continue

        seen_normals.append((w_unit, b_unit))
        lines.append((w, b))

    return lines


def partition_scene_svm(scene: Scene, C: float = 1.0) -> List[Partition]:
    """
    Partition the scene using SVM decision boundaries.

    Trains one-vs-one linear SVMs between each pair of robots, then uses the
    resulting maximum-margin separating lines to split the bounding box into
    convex cells.

    Args:
        scene: A discopygal Scene with robots and obstacles.
        C: SVM regularization parameter (soft-margin). Higher = harder margin.

    Returns:
        List of Partition objects covering the scene bounding box.
    """
    max_radius = max(robot.radius for robot in scene.robots)
    x_min, x_max, y_min, y_max = _compute_bounding_box(scene, padding=max_radius)

    # Start with the bounding box as a single cell
    bbox = shapely_box(x_min, y_min, x_max, y_max)
    cells: List[ShapelyPolygon] = [bbox]

    # Get SVM separating lines
    lines = _extract_svm_lines(scene, C)

    # Iteratively split cells by each SVM line
    for w, b in lines:
        w = np.array(w)
        new_cells = []
        for cell in cells:
            pieces = _split_polygon_by_line(cell, w, b)
            new_cells.extend(pieces)
        cells = new_cells

    # Convert to Partition objects
    partitions = []
    for cell in cells:
        cgal_poly = _shapely_to_cgal_polygon(cell)
        partitions.append(Partition(polygon=cgal_poly))

    return partitions