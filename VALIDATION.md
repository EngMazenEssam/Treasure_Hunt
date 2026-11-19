# Validation Guide

This document explains how to run `validate.py` and what it checks.

`validate.py` independently verifies the Phase I Treasure Hunt model and the four
uninformed search algorithms:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform-Cost Search (UCS)
- Iterative Deepening Search (IDS)

It uses only the Python standard library plus the project files `problem.py` and
`uninformed.py`. NumPy, Matplotlib, and Pillow are **not** needed for validation.

---

## 1. Requirements

- Python 3.8 or newer
- Project files in the same directory:
  - `validate.py`
  - `problem.py`
  - `uninformed.py`

No `pip install` is required for `validate.py`.

The plotting and animation scripts (`main.py`, `animate.py`) do need:

```text
numpy
matplotlib
pillow
```

---

## 2. How to run

Open a terminal in the project root, then run:

```bash
python validate.py
```

On Windows you can also use:

```powershell
py validate.py
```

On Linux/WSL with Python 3:

```bash
python3 validate.py
```

### Exit codes

| Exit code | Meaning |
|-----------|---------|
| `0` | All checks passed |
| `1` | At least one check failed |

The script prints `[PASS]`, `[FAIL]`, and `[INFO]` lines, followed by a summary.

### Save the output to a file

Linux/macOS/WSL:

```bash
python3 validate.py > results/validation_report.txt
```

Windows PowerShell:

```powershell
python validate.py | Tee-Object -FilePath results/validation_report.txt
```

---

## 3. What is validated

### 3.1 Problem model

`validate.py` checks that:

- Start, exit, key, and door positions exist.
- At least one treasure exists.
- The all-treasures mask matches the number of treasures.
- The theoretical state count equals:

```text
walkable cells × 2^(number of treasures) × 2 key states
```

- The reported reachable state count matches an independent graph traversal.
- The reachable count does not exceed the theoretical upper bound.
- The default map has between 1,000 and 10,000 reachable states, as required.
- Entering the key cell sets `has_key = 1`.
- Entering a treasure cell sets the correct mask bit.
- The locked door rejects an agent without the key.
- The door opens for an agent with the key.

### 3.2 Solution paths

For every algorithm, the validator checks:

- A solution was found.
- The path starts at the initial state.
- The number of actions equals the number of state transitions.
- Every action is legal according to `problem.get_successors`.
- Every action reaches the expected next state.
- The recomputed path cost equals the reported `path_cost`.
- The final state satisfies the goal test.

### 3.3 Metrics and animation trace

For every algorithm, the validator checks:

- `nodes_generated >= nodes_expanded >= 1`
- `max_frontier_size >= 1`
- `execution_time_ms >= 0`
- All expanded coordinates are inside the grid and not walls.
- The `trace` callback sees exactly `nodes_expanded` nodes.
- The last expanded state is the goal state.

This confirms that the data shown by `animate.py` matches the real search.

### 3.4 Theory-based expectations

On the default map:

- BFS returns a step-optimal path.
- IDS returns a step-optimal path.
- BFS and IDS use no more steps than DFS.
- UCS returns the minimum weighted cost.
- UCS cost is no worse than BFS, DFS, or IDS cost.

### 3.5 Weighted test map

The validator also runs a second, small map:

```text
PK0.M.E
.......
```

Here the shortest path by number of steps goes through mud, while a longer path
through grass is cheaper.

Expected behavior:

| Algorithm | Steps | Cost | Result |
|-----------|-------|------|--------|
| BFS | 6 | 10 | Step-optimal, **not** cost-optimal |
| UCS | 8 | 8 | Cost-optimal |
| IDS | 6 | 10 | Step-optimal, **not** cost-optimal |
| DFS | 14 | 18 | Valid but not optimal |

The DFS row is the current observed result; the validator does not require those
exact DFS numbers. It does require that UCS is no worse than DFS.

This test explicitly proves that BFS is not guaranteed to find the cheapest
weighted path. It only minimizes the number of steps.

---

## 4. Reference implementations used by the validator

To avoid simply trusting the algorithms being tested, `validate.py` contains
independent reference implementations:

- `reference_min_steps(problem)` — a separate BFS that computes the minimum
  number of actions.
- `reference_min_cost(problem)` — a separate Dijkstra that computes the minimum
  weighted cost.
- `reference_reachable_count(problem)` — a separate graph traversal that counts
  reachable states.

These use the problem model but do not reuse the code from `uninformed.py`.

---

## 5. Current expected result

On the default 15x15 map, the validator currently reports:

```text
reference minimum steps = 78
reference minimum cost  = 82

BFS: 78 steps, cost 82
DFS: 294 steps, cost 364
UCS: 78 steps, cost 82
IDS: 78 steps, cost 82
```

The final summary should look like:

```text
RESULT: 137/137 checks passed
All validation checks passed.
```

On the weighted map:

```text
reference minimum steps = 6
reference minimum cost  = 8

BFS: 6 steps, cost 10
UCS: 8 steps, cost 8
```

BFS matching UCS's cost on the large map is a property of that specific map,
not a general guarantee. The weighted map demonstrates the difference.

Execution times vary between runs depending on the machine and current load.
The validator checks that times are non-negative, not that they equal a fixed
value.

---

## 6. Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

You are probably running `main.py` or `animate.py`, not `validate.py`.
`validate.py` does not need NumPy.

- To validate only: `python validate.py`
- To run plots/animations: install the dependencies or use the Windows Python
  installation that already has NumPy and Matplotlib.

### The script reports a failed check

The final summary lists each failed check. Start with the first failure, because
later failures may be caused by it. Common causes:

- A change to the grid map changed the expected optimal cost or step count.
- A change to a search algorithm changed the path or metrics.
- A change to the goal test or transition model.

### A window does not open

`validate.py` is terminal-only. Windows are handled by `animate.py`, not the
validator.

---

## 7. When to run it

Run the validator:

- Before submitting Phase I.
- After changing the map, costs, or goal test.
- After changing any search algorithm.
- Before generating final plots and animations.

It is a quick correctness check and does not replace the report's complexity
analysis or the required video/contribution statement.
