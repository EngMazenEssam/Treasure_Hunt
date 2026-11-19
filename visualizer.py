"""
visualizer.py
Generates visual plots for Phase I:
    1. 2x2 grid heatmap of explored states and solution paths for BFS, DFS, UCS, IDS.
    2. 2x2 bar chart comparing Path Cost, Nodes Expanded, Max Frontier Size, and Runtime.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch


def plot_explored_and_paths(problem, results, output_path="results/explored_paths.png"):
    """Draws a 2x2 subplot of the 15x15 grid showing terrain, explored heatmap, and path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Encode terrain numerically for base layer
    # 0 = Grass/Entity (cost 1), 1 = Sand (cost 3), 2 = Mud (cost 5), 3 = Wall
    terrain_map = np.zeros((problem.rows, problem.cols), dtype=int)
    for r in range(problem.rows):
        for c in range(problem.cols):
            cell = problem.grid[r][c]
            if cell == "#":
                terrain_map[r, c] = 3
            elif cell == "S":
                terrain_map[r, c] = 1
            elif cell == "M":
                terrain_map[r, c] = 2
            else:
                terrain_map[r, c] = 0

    terrain_cmap = ListedColormap(["#eef7ea", "#f6e5b8", "#c9a684", "#2b2d42"])

    # Shared heat scale so the four heatmaps are directly comparable.
    max_heat = max(
        [max(res["visited_coords"].values(), default=1) for res in results] or [1]
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
    axes = axes.flatten()
    heat_image = None

    for idx, res in enumerate(results):
        ax = axes[idx]
        ax.imshow(terrain_map, cmap=terrain_cmap, vmin=0, vmax=3)

        # Overlay explored cells heatmap
        heat = np.zeros((problem.rows, problem.cols), dtype=float)
        for (r, c), count in res["visited_coords"].items():
            heat[r, c] = count
        masked_heat = np.ma.masked_where(heat == 0, heat)
        heat_image = ax.imshow(
            masked_heat,
            cmap="Blues",
            alpha=0.6,
            norm=LogNorm(vmin=1, vmax=max_heat),
        )

        # Draw grid lines
        ax.set_xticks(np.arange(-0.5, problem.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, problem.rows, 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate special entities (P, E, K, D, 0..3, S, M)
        for r in range(problem.rows):
            for c in range(problem.cols):
                cell = problem.grid[r][c]
                if cell not in {".", "#"}:
                    label = f"T{cell}" if cell in {"0", "1", "2", "3"} else cell
                    ax.text(
                        c,
                        r,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        fontweight="bold",
                        color="#111111",
                    )

        # Draw solution path
        if res["found"] and res["path_states"]:
            path_rows = [s[0] for s in res["path_states"]]
            path_cols = [s[1] for s in res["path_states"]]
            ax.plot(
                path_cols,
                path_rows,
                color="#e63946",
                linewidth=2.2,
                marker="o",
                markersize=3,
                label="Solution Path",
            )
            ax.plot(
                path_cols[0],
                path_rows[0],
                marker="s",
                markersize=8,
                color="#2a9d8f",
                markeredgecolor="black",
                linestyle="None",
                label="Start",
            )
            ax.plot(
                path_cols[-1],
                path_rows[-1],
                marker="*",
                markersize=13,
                color="#ffd166",
                markeredgecolor="black",
                linestyle="None",
                label="Goal",
            )

        steps = len(res["path_actions"])
        ax.set_title(
            f"{res['algorithm']} | Cost: {res['path_cost']} | Steps: {steps} | Expanded: {res['nodes_expanded']}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        "Phase I Uninformed Search: Explored States & Solution Paths (15x15 Treasure Hunt)",
        fontsize=14,
        fontweight="bold",
    )

    # Reserve a clean right-hand strip for the shared colorbar so it does not
    # overlap the four grid panels.
    fig.subplots_adjust(left=0.055, right=0.9, top=0.92, bottom=0.09, wspace=0.12, hspace=0.17)

    if heat_image is not None:
        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        cbar = fig.colorbar(heat_image, cax=cax, orientation="vertical")
        cbar.set_label("Times the cell was expanded (log scale)", fontsize=9)

    terrain_handles = [
        Patch(facecolor="#eef7ea", edgecolor="black", label="Grass (cost 1)"),
        Patch(facecolor="#f6e5b8", edgecolor="black", label="Sand (cost 3)"),
        Patch(facecolor="#c9a684", edgecolor="black", label="Mud (cost 5)"),
        Patch(facecolor="#2b2d42", edgecolor="black", label="Wall (blocked)"),
        plt.Line2D([], [], color="#e63946", linewidth=2.2, marker="o", markersize=3, label="Solution path"),
        plt.Line2D([], [], color="#2a9d8f", marker="s", markersize=8, linestyle="None", markeredgecolor="black", label="Start"),
        plt.Line2D([], [], color="#ffd166", marker="*", markersize=12, linestyle="None", markeredgecolor="black", label="Goal"),
    ]
    fig.legend(
        handles=terrain_handles,
        loc="lower center",
        ncol=7,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metrics_comparison(results, output_path="results/metrics_comparison.png"):
    """Generates a 2x2 comparison bar chart across all 4 uninformed algorithms."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    algos = [r["algorithm"] for r in results]
    colors = ["#457b9d", "#e76f51", "#2a9d8f", "#f4a261"]

    metrics = [
        ("path_cost", "Total Path Cost (Lower = Cheaper)", "Cost"),
        ("nodes_expanded", "Nodes Expanded (Time Effort)", "Nodes"),
        ("max_frontier_size", "Max Frontier Size (Peak Memory)", "Nodes"),
        ("execution_time_ms", "Execution Time (ms)", "Milliseconds"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, (key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        values = [r[key] for r in results]
        bars = ax.bar(algos, values, color=colors, edgecolor="#222222", width=0.55)

        # IDS is orders of magnitude slower/larger than the other algorithms, so a
        # linear axis would hide BFS/DFS/UCS. Switch to a log axis when needed.
        positive = [v for v in values if v > 0]
        use_log = bool(positive) and max(positive) / min(positive) > 20
        if use_log:
            ax.set_yscale("log")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{ylabel} (log scale)" if use_log else ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.5, which="both")

        for bar, val in zip(bars, values):
            label = f"{val:.2f}" if isinstance(val, float) else f"{val}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    plt.suptitle("Phase I Performance Comparison: BFS vs. DFS vs. UCS vs. IDS", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close(fig)