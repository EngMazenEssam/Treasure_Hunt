"""
animate.py
Real-time / instant simulation of the uninformed search algorithms on the
15x15 Treasure Hunt grid (CSAI 301 - Phase I).

What this script does
---------------------
1. Runs BFS / DFS / UCS / IDS on the same problem instance.
2. Records the order in which cells are scanned (first visit) plus the live
   number of expanded nodes and the current frontier size.
3. Renders the scan as an animation and, by default, writes it to a GIF so it
   can be dropped directly into the Phase I video. No GUI is required.
4. Can also render a 2x2 "race" animation that compares all algorithms side by
   side, which is ideal for the comparison part of the report/video.

Examples
--------
    python animate.py                              # all algorithms, GIFs
    python animate.py --algo bfs                   # one algorithm
    python animate.py --algo all --combined        # 2x2 comparison GIF
    python animate.py --algo all --combined --show # also try to open a window
    python animate.py --algo ids --interval 10 --frames 200

Only NumPy and Matplotlib are used (allowed by the project rules).
"""

import argparse
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from problem import TreasureHuntProblem
from uninformed import bfs, dfs, ucs, ids


ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "ucs": ucs,
    "ids": ids,
}

TERRAIN_CMAP = ListedColormap(["#eef7ea", "#f6e5b8", "#c9a684", "#2b2d42"])


class SearchTrace:
    """
    Callback passed to the search functions.

    The search functions call ``trace(state, expanded, frontier)`` every time a
    node is expanded. We keep:

    * ``first_visits``: the first time each grid cell is scanned, in scan order.
      This keeps the animation small (at most 147 frames) even for IDS, which
      expands more than 500,000 states.
    * ``cell_counts``: how often every cell was expanded (used for heat color).
    * ``max_expanded`` / ``max_frontier``: live counters for the title.
    """

    def __init__(self):
        self.first_visits = []  # (row, col, expanded, frontier, t_mask, has_key)
        self.cell_counts = {}
        self.max_expanded = 0
        self.max_frontier = 1
        self._seen_cells = set()

    def __call__(self, state, expanded, frontier):
        r, c, t_mask, has_key = state
        self.max_expanded = max(self.max_expanded, expanded)
        self.max_frontier = max(self.max_frontier, frontier)

        cell = (r, c)
        self.cell_counts[cell] = self.cell_counts.get(cell, 0) + 1

        if cell not in self._seen_cells:
            self._seen_cells.add(cell)
            self.first_visits.append((r, c, expanded, frontier, t_mask, has_key))


def run_search(problem, algorithm_name):
    """Runs one algorithm with tracing enabled and returns its result dict."""
    trace = SearchTrace()
    result = ALGORITHMS[algorithm_name](problem, trace=trace)
    result["trace"] = trace
    return result


def _terrain_array(problem):
    """Encodes the grid as 0=grass, 1=sand, 2=mud, 3=wall for imshow."""
    terrain = np.zeros((problem.rows, problem.cols), dtype=int)
    for r in range(problem.rows):
        for c in range(problem.cols):
            cell = problem.grid[r][c]
            if cell == "#":
                terrain[r, c] = 3
            elif cell == "S":
                terrain[r, c] = 1
            elif cell == "M":
                terrain[r, c] = 2
    return terrain


def _draw_static_cells(ax, problem):
    """Adds the entity labels (P, E, K, D, T0..T3, S, M) to a grid."""
    for r in range(problem.rows):
        for c in range(problem.cols):
            cell = problem.grid[r][c]
            if cell in {".", "#"}:
                continue
            label = f"T{cell}" if cell in {"0", "1", "2", "3"} else cell
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#111111",
            )


def _style_axes(ax, problem, title):
    ax.set_xticks(np.arange(-0.5, problem.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, problem.rows, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, fontsize=11, fontweight="bold")


def _heat_value(cell, cell_counts, max_cell_count):
    """Maps expansion frequency to a [0.3, 1.0] alpha/intensity value."""
    if max_cell_count <= 1:
        return 1.0
    return 0.3 + 0.7 * (cell_counts.get(cell, 1) / max_cell_count)


def _save_animation(anim, output_path, fps):
    """Saves an animation as GIF (default) or MP4 if ffmpeg is available."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    extension = os.path.splitext(output_path)[1].lower()

    if extension in {".mp4", ".mov"}:
        try:
            anim.save(output_path, writer="ffmpeg", fps=fps)
        except Exception as exc:  # ffmpeg is optional; fall back to GIF
            fallback = os.path.splitext(output_path)[0] + ".gif"
            print(f"[warn] MP4 export failed ({exc}). Writing GIF instead: {fallback}")
            anim.save(fallback, writer=PillowWriter(fps=fps))
            return fallback
    else:
        anim.save(output_path, writer=PillowWriter(fps=fps))

    return output_path


def animate_search(
    problem,
    result,
    output_path=None,
    interval_ms=40,
    target_frames=120,
    target_path_frames=80,
    fps=25,
):
    """
    Animates one algorithm: cells light up in scan order, then the final
    solution path is drawn step by step.
    """
    trace = result["trace"]
    events = trace.first_visits
    path_states = result["path_states"]

    search_skip = max(1, math.ceil(len(events) / target_frames)) if events else 1
    search_frames = max(1, math.ceil(len(events) / search_skip))

    path_skip = (
        max(1, math.ceil(len(path_states) / target_path_frames))
        if path_states
        else 1
    )
    path_frames = (
        max(1, math.ceil(len(path_states) / path_skip)) if path_states else 0
    )
    total_frames = search_frames + path_frames

    terrain = _terrain_array(problem)
    explored_grid = np.zeros((problem.rows, problem.cols), dtype=float)
    max_cell_count = max(trace.cell_counts.values(), default=1)

    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.imshow(terrain, cmap=TERRAIN_CMAP, vmin=0, vmax=3)
    heat_image = ax.imshow(
        np.ma.masked_where(explored_grid == 0, explored_grid),
        cmap="Blues",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    _draw_static_cells(ax, problem)
    _style_axes(ax, problem, result["algorithm"])

    (agent_dot,) = ax.plot(
        [], [], marker="o", markersize=10, color="#ff9f1c", markeredgecolor="black"
    )
    (path_line,) = ax.plot(
        [], [], color="#e63946", linewidth=2.5, marker="o", markersize=3.5
    )
    (start_dot,) = ax.plot(
        [], [], marker="s", markersize=8, color="#2a9d8f", markeredgecolor="black"
    )
    (goal_dot,) = ax.plot(
        [], [], marker="*", markersize=14, color="#ffd166", markeredgecolor="black"
    )

    if path_states:
        start_dot.set_data([path_states[0][1]], [path_states[0][0]])
        goal_dot.set_data([path_states[-1][1]], [path_states[-1][0]])

    last_event_index = 0

    def update(frame):
        nonlocal last_event_index

        if frame < search_frames:
            end_index = min((frame + 1) * search_skip, len(events))
            for idx in range(last_event_index, end_index):
                r, c, _, _, _, _ = events[idx]
                explored_grid[r, c] = _heat_value(
                    (r, c), trace.cell_counts, max_cell_count
                )
            last_event_index = end_index

            r, c, expanded, frontier, t_mask, has_key = events[end_index - 1]
            collected = bin(t_mask).count("1")
            heat_image.set_data(
                np.ma.masked_where(explored_grid == 0, explored_grid)
            )
            agent_dot.set_data([c], [r])
            ax.set_title(
                f"{result['algorithm']} scanning | Expanded: {expanded:,} | "
                f"Frontier: {frontier} | Treasures: {collected}/4 | "
                f"Key: {'Yes' if has_key else 'No'}",
                fontsize=10.5,
                fontweight="bold",
            )
            path_line.set_data([], [])
        else:
            # Make sure the whole scan is visible before drawing the path.
            if last_event_index < len(events):
                for idx in range(last_event_index, len(events)):
                    r, c, _, _, _, _ = events[idx]
                    explored_grid[r, c] = _heat_value(
                        (r, c), trace.cell_counts, max_cell_count
                    )
                last_event_index = len(events)
                heat_image.set_data(
                    np.ma.masked_where(explored_grid == 0, explored_grid)
                )

            path_index = min(
                (frame - search_frames + 1) * path_skip, len(path_states)
            )
            sub_path = path_states[:path_index]
            rows = [s[0] for s in sub_path]
            cols = [s[1] for s in sub_path]
            path_line.set_data(cols, rows)
            agent_dot.set_data([cols[-1]], [rows[-1]])
            ax.set_title(
                f"{result['algorithm']} GOAL FOUND | Steps: {len(path_states) - 1} | "
                f"Cost: {result['path_cost']} | Expanded: {result['nodes_expanded']:,}",
                fontsize=10.5,
                fontweight="bold",
            )

        return heat_image, agent_dot, path_line, start_dot, goal_dot

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()

    if output_path:
        saved = _save_animation(anim, output_path, fps)
        print(f"[saved] {result['algorithm'].upper()} animation -> {saved}")

    return anim


def animate_combined(
    problem,
    results,
    output_path=None,
    interval_ms=40,
    target_frames=120,
    target_path_frames=80,
    fps=25,
):
    """Renders a 2x2 side-by-side 'race' animation for all algorithms."""
    terrain = _terrain_array(problem)
    algorithms = [r["algorithm"] for r in results]

    search_skips = []
    path_skips = []
    for result in results:
        events_count = len(result["trace"].first_visits)
        path_count = len(result["path_states"])
        search_skips.append(max(1, math.ceil(events_count / target_frames)))
        path_skips.append(max(1, math.ceil(path_count / target_path_frames)))

    search_frames = max(
        max(1, math.ceil(len(r["trace"].first_visits) / skip))
        for r, skip in zip(results, search_skips)
    )
    path_frames = max(
        max(1, math.ceil(len(r["path_states"]) / skip))
        for r, skip in zip(results, path_skips)
    )
    total_frames = search_frames + path_frames

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    axes = axes.flatten()

    heat_images = []
    agent_dots = []
    path_lines = []
    explored_grids = []
    last_indices = [0] * len(results)

    for ax, result in zip(axes, results):
        trace = result["trace"]
        explored_grid = np.zeros((problem.rows, problem.cols), dtype=float)
        explored_grids.append(explored_grid)

        ax.imshow(terrain, cmap=TERRAIN_CMAP, vmin=0, vmax=3)
        heat_image = ax.imshow(
            np.ma.masked_where(explored_grid == 0, explored_grid),
            cmap="Blues",
            alpha=0.7,
            vmin=0,
            vmax=1,
        )
        heat_images.append(heat_image)

        _draw_static_cells(ax, problem)
        _style_axes(ax, problem, result["algorithm"])

        (agent_dot,) = ax.plot(
            [], [], marker="o", markersize=9, color="#ff9f1c", markeredgecolor="black"
        )
        (path_line,) = ax.plot(
            [], [], color="#e63946", linewidth=2.2, marker="o", markersize=3
        )
        agent_dots.append(agent_dot)
        path_lines.append(path_line)

    def update(frame):
        for i, (result, ax) in enumerate(zip(results, axes)):
            trace = result["trace"]
            events = trace.first_visits
            path_states = result["path_states"]
            explored_grid = explored_grids[i]

            if frame < search_frames:
                end_index = min((frame + 1) * search_skips[i], len(events))
                for idx in range(last_indices[i], end_index):
                    r, c, _, _, _, _ = events[idx]
                    explored_grid[r, c] = _heat_value(
                        (r, c),
                        trace.cell_counts,
                        max(trace.cell_counts.values(), default=1),
                    )
                last_indices[i] = end_index

                r, c, expanded, frontier, t_mask, has_key = events[end_index - 1]
                collected = bin(t_mask).count("1")
                heat_images[i].set_data(
                    np.ma.masked_where(explored_grid == 0, explored_grid)
                )
                agent_dots[i].set_data([c], [r])
                path_lines[i].set_data([], [])
                ax.set_title(
                    f"{result['algorithm']} | Expanded: {expanded:,} | "
                    f"Frontier: {frontier} | T: {collected}/4 | "
                    f"Key: {'Y' if has_key else 'N'}",
                    fontsize=9.5,
                    fontweight="bold",
                )
            else:
                if last_indices[i] < len(events):
                    for idx in range(last_indices[i], len(events)):
                        r, c, _, _, _, _ = events[idx]
                        explored_grid[r, c] = _heat_value(
                            (r, c),
                            trace.cell_counts,
                            max(trace.cell_counts.values(), default=1),
                        )
                    last_indices[i] = len(events)
                    heat_images[i].set_data(
                        np.ma.masked_where(explored_grid == 0, explored_grid)
                    )

                path_index = min(
                    (frame - search_frames + 1) * path_skips[i],
                    len(path_states),
                )
                sub_path = path_states[:path_index]
                rows = [s[0] for s in sub_path]
                cols = [s[1] for s in sub_path]
                path_lines[i].set_data(cols, rows)
                agent_dots[i].set_data([cols[-1]], [rows[-1]])
                ax.set_title(
                    f"{result['algorithm']} | Steps: {len(path_states) - 1} | "
                    f"Cost: {result['path_cost']} | Expanded: {result['nodes_expanded']:,}",
                    fontsize=9.5,
                    fontweight="bold",
                )

        return heat_images + agent_dots + path_lines

    fig.suptitle(
        "Phase I Treasure Hunt - Real-Time Search Comparison",
        fontsize=15,
        fontweight="bold",
    )
    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        saved = _save_animation(anim, output_path, fps)
        print(f"[saved] combined animation -> {saved}")

    return anim


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time simulation of BFS/DFS/UCS/IDS on the Treasure Hunt grid."
    )
    parser.add_argument(
        "--algo",
        choices=["bfs", "dfs", "ucs", "ids", "all"],
        default="all",
        help="Algorithm to animate (default: all).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Render one 2x2 comparison animation instead of one file per algorithm.",
    )
    parser.add_argument(
        "--save-dir",
        default="results/animations",
        help="Directory for generated GIF/MP4 files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=40,
        help="Milliseconds between displayed frames (default: 40).",
    )
    parser.add_argument(
        "--fps", type=int, default=25, help="Frames per second in the saved file."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Target number of search frames per algorithm (default: 120).",
    )
    parser.add_argument(
        "--path-frames",
        type=int,
        default=80,
        help="Target number of frames used to draw the final path (default: 80).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also try to open an interactive window (requires a GUI backend).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Saving GIFs/MP4 does not need a display server, so use Agg by default.
    # --show keeps the normal GUI backend (set MPLBACKEND to choose one).
    if not args.show and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg", force=True)

    problem = TreasureHuntProblem()

    names = list(ALGORITHMS) if args.algo == "all" else [args.algo]
    results = [run_search(problem, name) for name in names]

    print("=" * 68)
    print("REAL-TIME SEARCH SIMULATION")
    print("=" * 68)
    for result in results:
        print(
            f"{result['algorithm']:<4} | Path cost: {result['path_cost']:<4} | "
            f"Steps: {len(result['path_actions']):<4} | "
            f"Expanded: {result['nodes_expanded']:,} | "
            f"Unique cells scanned: {len(result['trace'].first_visits)}"
        )

    os.makedirs(args.save_dir, exist_ok=True)

    if args.combined and len(results) > 1:
        output_path = os.path.join(args.save_dir, "all_algorithms.gif")
        anim = animate_combined(
            problem,
            results,
            output_path=output_path,
            interval_ms=args.interval,
            target_frames=args.frames,
            target_path_frames=args.path_frames,
            fps=args.fps,
        )
    else:
        anim = None
        for result in results:
            output_path = os.path.join(
                args.save_dir, f"{result['algorithm'].lower()}_search.gif"
            )
            anim = animate_search(
                problem,
                result,
                output_path=output_path,
                interval_ms=args.interval,
                target_frames=args.frames,
                target_path_frames=args.path_frames,
                fps=args.fps,
            )

    if args.show and anim is not None:
        plt.show()


if __name__ == "__main__":
    main()
