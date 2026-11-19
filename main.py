"""
main.py
Entry point for CSAI 301 Phase I experiments.
Runs BFS, DFS, UCS, and IDS on the Treasure Hunt Problem, prints metrics,
exports results to CSV, and generates visualization charts.
"""

import csv
import os
from problem import TreasureHuntProblem
from uninformed import bfs, dfs, ucs, ids
from visualizer import plot_explored_and_paths, plot_metrics_comparison


def export_csv(results, filepath="results/phase1_metrics.csv"):
    """Writes experimental metrics to a CSV file for report tables."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = [
        "Algorithm",
        "Found",
        "Steps",
        "Path_Cost",
        "Nodes_Generated",
        "Nodes_Expanded",
        "Max_Frontier_Size",
        "Execution_Time_ms",
    ]
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Algorithm": r["algorithm"],
                    "Found": r["found"],
                    "Steps": len(r["path_actions"]),
                    "Path_Cost": r["path_cost"],
                    "Nodes_Generated": r["nodes_generated"],
                    "Nodes_Expanded": r["nodes_expanded"],
                    "Max_Frontier_Size": r["max_frontier_size"],
                    "Execution_Time_ms": round(r["execution_time_ms"], 4),
                }
            )


def main():
    problem = TreasureHuntProblem()
    stats = problem.get_state_space_stats()

    print("=" * 72)
    print("CSAI 301 - PHASE I: TREASURE HUNT PROBLEM MODEL & STATE SPACE")
    print("=" * 72)
    print(f"Grid Dimensions      : {stats['rows']} x {stats['cols']} ({stats['total_cells']} total cells)")
    print(f"Walkable Cells       : {stats['walkable_cells']} (Walls: {stats['wall_cells']})")
    print(
        f"Theoretical Upper Bound: {stats['theoretical_states']} states "
        f"({stats['walkable_cells']} cells * 2^{stats['treasure_count']} masks * 2 key states)"
    )
    print(f"Exact Reachable States : {stats['reachable_states']} states (target range: 1,000-10,000)")
    print(f"Initial State        : {problem.get_initial_state()}")
    print(f"Exit Position        : {problem.exit_pos} (Door at {problem.door_pos}, Key at {problem.key_pos})")
    print(f"Treasures            : {problem.treasures}")
    print("=" * 72)

    algorithms = [bfs, dfs, ucs, ids]
    results = [algo(problem) for algo in algorithms]

    header = f"{'Algo':<6} | {'Found':<5} | {'Steps':<5} | {'Cost':<5} | {'Generated':<9} | {'Expanded':<8} | {'MaxFront':<8} | {'Time (ms)':<9}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['algorithm']:<6} | "
            f"{str(r['found']):<5} | "
            f"{len(r['path_actions']):<5} | "
            f"{r['path_cost']:<5} | "
            f"{r['nodes_generated']:<9} | "
            f"{r['nodes_expanded']:<8} | "
            f"{r['max_frontier_size']:<8} | "
            f"{r['execution_time_ms']:<9.3f}"
        )

    optimal_cost = min(r["path_cost"] for r in results if r["found"])
    print(
        "\nNote: BFS and IDS are optimal in number of STEPS, not in weighted cost. "
        "UCS is the cost-optimal algorithm."
    )
    print(
        f"      Shortest weighted cost on this map = {optimal_cost}. "
        "Any algorithm reporting that cost here matches UCS on this instance only."
    )

    export_csv(results, "results/phase1_metrics.csv")
    plot_explored_and_paths(problem, results, "results/explored_paths.png")
    plot_metrics_comparison(results, "results/metrics_comparison.png")

    print("\nSaved outputs to ./results/:")
    print("  - results/phase1_metrics.csv")
    print("  - results/explored_paths.png")
    print("  - results/metrics_comparison.png")


if __name__ == "__main__":
    main()