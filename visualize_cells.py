import argparse
import os
import subprocess
import sys

# Force Agg before pyplot imports: Qt/Tk backends deadlock inside
# solver_viewer's event loop. --show opens the saved PNG instead.
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402

from discopygal.solvers_infra import Scene

from high_level_graph import HighLevelGraph, build_high_level_graph
from scene_partitioning import partition_free_space_grid


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "visualizations")


def _poly_xy(poly):
    """(xs, ys) lists of polygon vertex coordinates."""
    xs = [v.x().to_double() for v in poly.vertices()]
    ys = [v.y().to_double() for v in poly.vertices()]
    return xs, ys


def _obstacle_xy(obs):
    """(xs, ys) of an ObstaclePolygon's outline."""
    return _poly_xy(obs.poly)


def _centroid(xs, ys):
    """Mean of xs, mean of ys."""
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bounds(scene):
    """Padded scene bounding box across obstacles, starts, and goals."""
    xs, ys = [], []
    for o in scene.obstacles:
        ox, oy = _obstacle_xy(o)
        xs.extend(ox)
        ys.extend(oy)
    for r in scene.robots:
        xs.append(r.start.x().to_double())
        ys.append(r.start.y().to_double())
        xs.append(r.end.x().to_double())
        ys.append(r.end.y().to_double())
    if not xs:
        return -10, 10, -10, 10
    pad = 1.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _draw_cells_layer(ax, partitions, show_cell_ids: bool) -> None:
    """Filled cell polygons + optional id/density labels."""
    cell_patches = []
    cell_colors = []
    cmap = plt.get_cmap("tab20")
    for ci, part in enumerate(partitions):
        xs, ys = _poly_xy(part.polygon)
        cell_patches.append(mpatches.Polygon(list(zip(xs, ys)), closed=True))
        cell_colors.append(cmap(ci % 20))
    ax.add_collection(PatchCollection(
        cell_patches, facecolor=cell_colors, edgecolor="#444",
        linewidths=0.8, alpha=0.25,
    ))
    if show_cell_ids:
        for ci, part in enumerate(partitions):
            xs, ys = _poly_xy(part.polygon)
            cx, cy = _centroid(xs, ys)
            ax.text(
                cx, cy, f"{ci}\n(k={part.density})",
                fontsize=7, ha="center", va="center", color="#222",
            )


def _draw_ports_layer(ax, hlg: HighLevelGraph) -> None:
    """One dot per port point, plus a thin line connecting the two
    points of each port pair."""
    port_points: dict = {}
    for ci, port_map in hlg.cell_boundary_ports.items():
        for port_id, (x, y) in port_map.items():
            port_points.setdefault(port_id, []).append((ci, x, y))
    for entries in port_points.values():
        if len(entries) >= 2:
            (_, x1, y1), (_, x2, y2) = entries[0], entries[1]
            ax.plot(
                [x1, x2], [y1, y2],
                color="#1f77b4", linewidth=0.7, alpha=0.6, zorder=4,
            )
        for _, x, y in entries:
            ax.plot(x, y, "o", color="#1f77b4", markersize=3, zorder=5)


def _draw_robots_layer(ax, robots, robot_radius: float) -> None:
    """Green disc at each start, red square at each goal, dotted line between."""
    for ri, r in enumerate(robots):
        sx = r.start.x().to_double()
        sy = r.start.y().to_double()
        gx = r.end.x().to_double()
        gy = r.end.y().to_double()
        ax.add_patch(mpatches.Circle(
            (sx, sy), robot_radius,
            facecolor="#2ca02c", edgecolor="#114d11",
            alpha=0.55, linewidth=0.8, zorder=6,
        ))
        ax.add_patch(mpatches.Rectangle(
            (gx - robot_radius, gy - robot_radius),
            2 * robot_radius, 2 * robot_radius,
            facecolor="#d62728", edgecolor="#661111",
            alpha=0.55, linewidth=0.8, zorder=6,
        ))
        ax.plot(
            [sx, gx], [sy, gy],
            linestyle=":", color="#888", linewidth=0.5, zorder=2,
        )
        ax.text(
            sx, sy, str(ri),
            fontsize=6, ha="center", va="center",
            color="white", zorder=7,
        )


def draw_cells(
    scene: Scene,
    partitions: list,
    hlg: HighLevelGraph,
    robot_radius: float,
    save_path: str | None = None,
    title: str | None = None,
    show_ports: bool = True,
    show_cell_ids: bool = True,
    show_robots: bool = True,
    show_interactive: bool = False,
    verbose: bool = False,
) -> str | None:
    """Render the decomposition and save a PNG.

    :param scene: discopygal scene.
    :param partitions: cells.
    :param hlg: high-level graph.
    :param robot_radius: disc radius r.
    :param save_path: target PNG path, or None for a timestamped default
        under visualizations/.
    :param title: figure title, or None for an auto-generated one.
    :param show_ports: draw port points.
    :param show_cell_ids: label each cell with its id and density.
    :param show_robots: draw start/goal markers.
    :param show_interactive: try to open the saved PNG with the platform
        viewer (Agg backend can't pop a matplotlib window directly).
    :param verbose: print the saved-to path.
    :return: the saved path.
    """
    num_ports = sum(len(p) for p in hlg.cell_boundary_ports.values()) // 2

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")

    _draw_cells_layer(ax, partitions, show_cell_ids)

    for o in scene.obstacles:
        ox, oy = _obstacle_xy(o)
        ax.fill(ox, oy, color="#222", alpha=0.9, zorder=3)

    if show_ports:
        _draw_ports_layer(ax, hlg)

    if show_robots:
        _draw_robots_layer(ax, scene.robots, robot_radius)

    xmin, xmax, ymin, ymax = _bounds(scene)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(title or (
        f"{len(partitions)} cells, {num_ports} ports, "
        f"{len(scene.robots)} robots"
    ))
    ax.grid(True, alpha=0.2)

    legend_handles: list = [
        mpatches.Patch(color="#222", alpha=0.9, label="obstacle"),
        mpatches.Patch(color="#1f77b4", alpha=0.5, label="cell"),
        mpatches.Patch(color="#2ca02c", alpha=0.55, label="start"),
        mpatches.Patch(color="#d62728", alpha=0.55, label="goal"),
    ]
    if show_ports:
        legend_handles.append(plt.Line2D(
            [0], [0], marker="o", color="#1f77b4",
            linestyle="-", markersize=4, label="port point",
        ))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is None:
        from datetime import datetime
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out: str = os.path.join(DEFAULT_OUTPUT_DIR, f"cells_{stamp}.png")
    else:
        out = save_path
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    if verbose:
        print(f"saved to {out}")
    plt.close(fig)

    if show_interactive:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", out])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", out])
            elif sys.platform == "win32":
                os.startfile(out)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"could not open viewer: {exc}")

    return out


def visualize(
    scene_path: str,
    density: int = 100,
    show_ports: bool = True,
    show_cell_ids: bool = True,
    show_robots: bool = True,
    save_path: str | None = None,
    show_interactive: bool = False,
) -> None:
    """Run the decomposition + HLG build for a scene file and render a PNG.

    :param scene_path: path to a discopygal scene JSON.
    :param density: max_cell_density for the partitioner.
    :param show_ports: see draw_cells.
    :param show_cell_ids: see draw_cells.
    :param show_robots: see draw_cells.
    :param save_path: target PNG path; auto-generated if None.
    :param show_interactive: open the PNG with the platform viewer.
    """
    scene = Scene.from_file(scene_path)
    robots = scene.robots
    if not robots:
        raise SystemExit(f"{scene_path} has no robots")
    robot_radius = robots[0].radius.to_double()

    partitions, _ = partition_free_space_grid(
        scene, robot_radius, max_cell_density=density,
    )

    robot_starts = [
        (r.start.x().to_double(), r.start.y().to_double()) for r in robots
    ]
    robot_goals = [
        (r.end.x().to_double(), r.end.y().to_double()) for r in robots
    ]
    hlg = build_high_level_graph(
        partitions, robot_starts, robot_goals, robot_radius,
    )

    num_ports = sum(len(p) for p in hlg.cell_boundary_ports.values()) // 2
    print(
        f"{os.path.basename(scene_path)}: {len(partitions)} cells, "
        f"{hlg.graph.number_of_nodes()} hlg nodes, "
        f"{hlg.graph.number_of_edges()} hlg edges, {num_ports} ports"
    )

    if save_path is None and not show_interactive:
        stem = os.path.splitext(os.path.basename(scene_path))[0]
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(
            DEFAULT_OUTPUT_DIR, f"{stem}_d{density}.png",
        )

    title = (
        f"{os.path.basename(scene_path)} -- density={density}, "
        f"{len(partitions)} cells, {num_ports} ports, {len(robots)} robots"
    )
    draw_cells(
        scene, partitions, hlg, robot_radius,
        save_path=save_path,
        title=title,
        show_ports=show_ports,
        show_cell_ids=show_cell_ids,
        show_robots=show_robots,
        show_interactive=show_interactive,
        verbose=True,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Draw cells, ports, and robots for a Discopygal scene."
    )
    p.add_argument("scene", help="Path to a Discopygal scene JSON")
    p.add_argument(
        "--density", type=int, default=100,
        help="max_cell_density passed to the partitioner (default: 100)",
    )
    p.add_argument("--no-ports", action="store_true", help="Hide ports")
    p.add_argument("--no-ids", action="store_true", help="Hide cell ids")
    p.add_argument("--no-robots", action="store_true", help="Hide start/goal")
    p.add_argument(
        "--save", metavar="PATH",
        help="Save PNG to PATH (default: visualizations/<scene>_d<density>.png)",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Open the saved PNG with the platform viewer",
    )
    args = p.parse_args(argv)

    visualize(
        args.scene,
        density=args.density,
        show_ports=not args.no_ports,
        show_cell_ids=not args.no_ids,
        show_robots=not args.no_robots,
        save_path=args.save,
        show_interactive=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
