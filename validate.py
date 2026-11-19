"""
validate.py
Independent validation of the Phase I Treasure Hunt model and the four
uninformed search algorithms (BFS, DFS, UCS, IDS).

Run:
    python validate.py

The script checks:
    1. Problem model (entities, costs, key/door/treasure transitions, state count).
    2. Each returned solution path (actions, states, cost, goal test).
    3. Each algorithm's reported metrics and trace hook.
    4. Theory-based expectations (BFS/IDS step-optimal, UCS cost-optimal).
    5. A weighted test map where the shortest-step path is NOT the cheapest path,
       which exposes the difference between BFS/IDS and UCS.

Exit code 0 means every check passed.
"""

import heapq
import sys
from collections import deque

from problem import TreasureHuntProblem
from uninformed import bfs, dfs, ucs, ids


ALGORITHMS = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("IDS", ids)]

WEIGHTED_MAP = [
    "PK0.M.E",
    ".......",
]


class Report:
    """Collects PASS/FAIL checks and prints a final summary."""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failures = []

    def section(self, title):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)

    def check(self, condition, message):
        self.total += 1
        if condition:
            self.passed += 1
            print(f"[PASS] {message}")
        else:
            self.failures.append(message)
            print(f"[FAIL] {message}")
        return bool(condition)

    def info(self, message):
        print(f"[INFO] {message}")

    def summary(self):
        print("\n" + "=" * 72)
        print(f"RESULT: {self.passed}/{self.total} checks passed")
        if self.failures:
            print("Failed checks:")
            for failure in self.failures:
                print(f"  - {failure}")
        else:
            print("All validation checks passed.")
        print("=" * 72)


class TraceCounter:
    """Minimal trace callback used to verify the animation instrumentation."""

    def __init__(self):
        self.count = 0
        self.last_state = None

    def __call__(self, state, expanded, frontier):
        self.count += 1
        self.last_state = state


def reference_min_steps(problem):
    """Independent BFS reference for the minimum number of actions."""
    start = problem.get_initial_state()
    queue = deque([(start, 0)])
    seen = {start}

    while queue:
        state, depth = queue.popleft()
        if problem.is_goal(state):
            return depth
        for _, next_state, _ in problem.get_successors(state):
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, depth + 1))
    return None


def reference_min_cost(problem):
    """Independent Dijkstra reference for the minimum weighted path cost."""
    start = problem.get_initial_state()
    heap = [(0, 0, start)]
    counter = 1
    best = {start: 0}

    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > best.get(state, float("inf")):
            continue
        if problem.is_goal(state):
            return cost
        for _, next_state, step_cost in problem.get_successors(state):
            new_cost = cost + step_cost
            if new_cost < best.get(next_state, float("inf")):
                best[next_state] = new_cost
                heapq.heappush(heap, (new_cost, counter, next_state))
                counter += 1
    return None


def reference_reachable_count(problem):
    """Independent traversal that counts every state reachable from the start."""
    start = problem.get_initial_state()
    seen = {start}
    queue = deque([start])

    while queue:
        state = queue.popleft()
        for _, next_state, _ in problem.get_successors(state):
            if next_state not in seen:
                seen.add(next_state)
                queue.append(next_state)
    return len(seen)


def find_walkable_neighbour(problem, position):
    """Returns one walkable neighbour of a cell, or None."""
    r, c = position
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < problem.rows and 0 <= nc < problem.cols:
            if problem.grid[nr][nc] != "#":
                return (nr, nc)
    return None


def run_algorithm(problem, algorithm_function):
    """Runs one algorithm while verifying that its trace callback is complete."""
    trace = TraceCounter()
    result = algorithm_function(problem, trace=trace)
    return result, trace


def validate_path(problem, result, trace, report):
    """Validates one result dictionary against the problem model."""
    algorithm = result["algorithm"]

    if not report.check(result["found"], f"{algorithm}: found a solution"):
        return

    initial_state = problem.get_initial_state()
    path_states = result["path_states"]
    path_actions = result["path_actions"]

    report.check(
        bool(path_states) and path_states[0] == initial_state,
        f"{algorithm}: path starts at the initial state",
    )
    report.check(
        len(path_actions) == len(path_states) - 1,
        f"{algorithm}: number of actions matches number of transitions",
    )

    state = initial_state
    recomputed_cost = 0
    path_is_valid = True
    reason = "ok"

    for step, (action, expected_state) in enumerate(
        zip(path_actions, path_states[1:])
    ):
        matches = [
            (next_state, cost)
            for next_action, next_state, cost in problem.get_successors(state)
            if next_action == action
        ]
        if len(matches) != 1:
            path_is_valid = False
            reason = f"action {action} is not valid at step {step}"
            break

        next_state, cost = matches[0]
        if next_state != expected_state:
            path_is_valid = False
            reason = f"action {action} at step {step} produced a different state"
            break

        state = next_state
        recomputed_cost += cost

    report.check(path_is_valid, f"{algorithm}: every path action is legal ({reason})")
    if path_is_valid:
        report.check(
            recomputed_cost == result["path_cost"],
            f"{algorithm}: path cost recomputes to {recomputed_cost}",
        )

    report.check(
        problem.is_goal(path_states[-1]),
        f"{algorithm}: final state satisfies the goal test",
    )

    report.check(
        result["nodes_generated"] >= result["nodes_expanded"] >= 1,
        f"{algorithm}: generated >= expanded >= 1",
    )
    report.check(
        result["max_frontier_size"] >= 1,
        f"{algorithm}: max frontier size is positive",
    )
    report.check(
        result["execution_time_ms"] >= 0,
        f"{algorithm}: execution time is non-negative",
    )

    bad_coords = []
    for r, c in result["visited_coords"]:
        inside = 0 <= r < problem.rows and 0 <= c < problem.cols
        if not inside or problem.grid[r][c] == "#":
            bad_coords.append((r, c))
    report.check(
        not bad_coords,
        f"{algorithm}: expanded coordinates are walkable ({len(bad_coords)} invalid)",
    )

    report.check(
        trace.count == result["nodes_expanded"],
        f"{algorithm}: trace callback sees every expanded node",
    )
    report.check(
        trace.last_state == path_states[-1],
        f"{algorithm}: last expanded state is the goal state",
    )


def validate_model(report, problem, label, require_door=True):
    """Checks model invariants that both test maps must satisfy."""
    report.section(f"MODEL: {label}")

    report.check(problem.start_pos is not None, "start position exists")
    report.check(problem.exit_pos is not None, "exit position exists")
    report.check(problem.key_pos is not None, "key position exists")
    if require_door:
        report.check(problem.door_pos is not None, "door position exists")
    report.check(len(problem.treasures) > 0, "at least one treasure exists")
    report.check(
        problem.all_treasures_mask == (1 << len(problem.treasures)) - 1,
        "treasure mask covers every treasure",
    )

    stats = problem.get_state_space_stats()
    expected_theoretical = (
        stats["walkable_cells"]
        * (1 << len(problem.treasures))
        * 2
    )
    report.check(
        stats["theoretical_states"] == expected_theoretical,
        "theoretical state count matches cells x treasure masks x key states",
    )

    reachable = reference_reachable_count(problem)
    report.check(
        stats["reachable_states"] == reachable,
        f"reported reachable states ({stats['reachable_states']}) match independent count ({reachable})",
    )
    report.check(
        reachable <= stats["theoretical_states"],
        "reachable states do not exceed the theoretical upper bound",
    )

    return stats


def validate_default_map(report):
    report.section("TEST 1: DEFAULT 15x15 MAP")
    problem = TreasureHuntProblem()
    stats = validate_model(report, problem, "default map")

    report.check(
        1000 <= stats["reachable_states"] <= 10000,
        f"default state space ({stats['reachable_states']}) is in the 1,000-10,000 target range",
    )
    report.check(
        problem.get_initial_state() == (0, 0, 0, 0),
        "initial state is (0, 0, mask=0, key=0)",
    )

    key_r, key_c = problem.key_pos
    key_neighbour = find_walkable_neighbour(problem, problem.key_pos)
    report.check(key_neighbour is not None, "key has a walkable neighbour")
    if key_neighbour is not None:
        before = (key_neighbour[0], key_neighbour[1], 0, 0)
        after_key = [
            next_state
            for _, next_state, _ in problem.get_successors(before)
            if (next_state[0], next_state[1]) == (key_r, key_c)
        ]
        report.check(
            bool(after_key) and all(s[3] == 1 for s in after_key),
            "entering the key cell sets has_key=1",
        )

    treasure_pos, treasure_bit = next(iter(problem.treasures.items()))
    treasure_neighbour = find_walkable_neighbour(problem, treasure_pos)
    report.check(treasure_neighbour is not None, "treasure has a walkable neighbour")
    if treasure_neighbour is not None:
        before = (treasure_neighbour[0], treasure_neighbour[1], 0, 0)
        after_treasure = [
            next_state
            for _, next_state, _ in problem.get_successors(before)
            if (next_state[0], next_state[1]) == treasure_pos
        ]
        report.check(
            bool(after_treasure)
            and all(s[2] & (1 << treasure_bit) for s in after_treasure),
            "entering a treasure cell sets its mask bit",
        )

    door_r, door_c = problem.door_pos
    door_neighbour = find_walkable_neighbour(problem, problem.door_pos)
    report.check(door_neighbour is not None, "door has a walkable neighbour")
    if door_neighbour is not None:
        without_key = (
            door_neighbour[0],
            door_neighbour[1],
            problem.all_treasures_mask,
            0,
        )
        into_door_without_key = [
            next_state
            for _, next_state, _ in problem.get_successors(without_key)
            if (next_state[0], next_state[1]) == (door_r, door_c)
        ]
        report.check(
            not into_door_without_key,
            "locked door rejects an agent without the key",
        )

        with_key = (
            door_neighbour[0],
            door_neighbour[1],
            problem.all_treasures_mask,
            1,
        )
        into_door_with_key = [
            (next_state, cost)
            for _, next_state, cost in problem.get_successors(with_key)
            if (next_state[0], next_state[1]) == (door_r, door_c)
        ]
        report.check(
            bool(into_door_with_key)
            and all(s[3] == 1 for s, _ in into_door_with_key),
            "door opens for an agent with the key",
        )

    results = {}
    for name, algorithm_function in ALGORITHMS:
        result, trace = run_algorithm(problem, algorithm_function)
        results[name] = result
        validate_path(problem, result, trace, report)

    min_steps = reference_min_steps(problem)
    min_cost = reference_min_cost(problem)

    report.info(f"reference minimum steps = {min_steps}")
    report.info(f"reference minimum cost  = {min_cost}")

    report.check(
        len(results["BFS"]["path_actions"]) == min_steps,
        f"BFS returns a step-optimal path ({len(results['BFS']['path_actions'])} steps)",
    )
    report.check(
        len(results["IDS"]["path_actions"]) == min_steps,
        f"IDS returns a step-optimal path ({len(results['IDS']['path_actions'])} steps)",
    )
    report.check(
        results["UCS"]["path_cost"] == min_cost,
        f"UCS returns a cost-optimal path ({results['UCS']['path_cost']} cost)",
    )
    report.check(
        len(results["BFS"]["path_actions"])
        <= len(results["DFS"]["path_actions"]),
        "BFS uses no more steps than DFS",
    )
    report.check(
        len(results["IDS"]["path_actions"])
        <= len(results["DFS"]["path_actions"]),
        "IDS uses no more steps than DFS",
    )

    for name in ["BFS", "DFS", "IDS"]:
        report.check(
            results["UCS"]["path_cost"] <= results[name]["path_cost"],
            f"UCS cost {results['UCS']['path_cost']} <= {name} cost {results[name]['path_cost']}",
        )

    if results["BFS"]["path_cost"] == min_cost:
        report.info(
            "BFS also matched the optimal cost on this map, but BFS is not "
            "cost-optimal in general (see Test 2)."
        )
    report.info(
        f"BFS expanded {results['BFS']['nodes_expanded']:,} nodes, "
        f"max frontier {results['BFS']['max_frontier_size']}"
    )
    report.info(
        f"IDS expanded {results['IDS']['nodes_expanded']:,} nodes, "
        f"max frontier {results['IDS']['max_frontier_size']}"
    )
    report.info(
        f"DFS expanded {results['DFS']['nodes_expanded']:,} nodes, "
        f"max frontier {results['DFS']['max_frontier_size']}"
    )

    return results


def validate_weighted_map(report):
    report.section("TEST 2: WEIGHTED MAP (SHORTEST STEPS != CHEAPEST PATH)")
    problem = TreasureHuntProblem(WEIGHTED_MAP)
    validate_model(report, problem, "weighted test map", require_door=False)

    results = {}
    for name, algorithm_function in ALGORITHMS:
        result, trace = run_algorithm(problem, algorithm_function)
        results[name] = result
        validate_path(problem, result, trace, report)

    min_steps = reference_min_steps(problem)
    min_cost = reference_min_cost(problem)

    report.info(f"reference minimum steps = {min_steps}")
    report.info(f"reference minimum cost  = {min_cost}")
    report.info(
        f"BFS: {len(results['BFS']['path_actions'])} steps, "
        f"cost {results['BFS']['path_cost']}"
    )
    report.info(
        f"UCS: {len(results['UCS']['path_actions'])} steps, "
        f"cost {results['UCS']['path_cost']}"
    )

    report.check(
        len(results["BFS"]["path_actions"]) == min_steps,
        "BFS finds the minimum-step path on the weighted map",
    )
    report.check(
        len(results["IDS"]["path_actions"]) == min_steps,
        "IDS finds the minimum-step path on the weighted map",
    )
    report.check(
        results["UCS"]["path_cost"] == min_cost,
        "UCS finds the minimum-cost path on the weighted map",
    )
    report.check(
        results["BFS"]["path_cost"] > results["UCS"]["path_cost"],
        "BFS is NOT cost-optimal on the weighted map (expected)",
    )
    report.check(
        len(results["BFS"]["path_actions"])
        < len(results["UCS"]["path_actions"]),
        "BFS uses fewer steps than UCS on the weighted map (expected)",
    )
    report.check(
        results["UCS"]["path_cost"] <= results["DFS"]["path_cost"],
        "UCS cost is no worse than DFS on the weighted map",
    )
    report.check(
        results["UCS"]["path_cost"] <= results["IDS"]["path_cost"],
        "UCS cost is no worse than IDS on the weighted map",
    )

    return results

def validate_algorithm_signatures(report):
    """Verifies internal behavioral signatures (not just final output) of each algorithm."""
    report.section("TEST 3: INTERNAL ALGORITHM BEHAVIOR SIGNATURES")
    problem = TreasureHuntProblem()
    start = problem.get_initial_state()

    # 1. BFS Signature: expands nodes in strictly non-decreasing step-depth order
    bfs_depths = []
    # Compute exact shortest step-distance from start for every reachable state
    dist_map = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        for _, ns, _ in problem.get_successors(s):
            if ns not in dist_map:
                dist_map[ns] = dist_map[s] + 1
                q.append(ns)

    bfs(problem, trace=lambda s, exp, front: bfs_depths.append(dist_map[s]))
    is_monotone_depth = all(
        bfs_depths[i] <= bfs_depths[i + 1] for i in range(len(bfs_depths) - 1)
    )
    report.check(
        is_monotone_depth,
        "BFS signature: expands states in strictly non-decreasing depth order (FIFO wave)",
    )

    # 2. UCS Signature: expands states in non-decreasing cumulative g(n) cost order
    ucs_costs = []
    best_g = {start: 0}
    pq = [(0, 0, start)]
    c = 1
    while pq:
        g, _, s = heapq.heappop(pq)
        if g > best_g.get(s, float("inf")):
            continue
        for _, ns, sc in problem.get_successors(s):
            if g + sc < best_g.get(ns, float("inf")):
                best_g[ns] = g + sc
                heapq.heappush(pq, (g + sc, c, ns))
                c += 1

    ucs(problem, trace=lambda s, exp, front: ucs_costs.append(best_g[s]))
    is_monotone_cost = all(
        ucs_costs[i] <= ucs_costs[i + 1] for i in range(len(ucs_costs) - 1)
    )
    report.check(
        is_monotone_cost,
        "UCS signature: expands states in non-decreasing path cost g(n) order (min-heap)",
    )

    # 3. IDS Signature: repeatedly restarts from initial_state (one restart per depth limit)
    #    and keeps max_frontier_size strictly smaller than BFS
    ids_starts = 0

    def ids_trace(s, exp, front):
        nonlocal ids_starts
        if s == start:
            ids_starts += 1

    ids_res = ids(problem, trace=ids_trace)
    bfs_res = bfs(problem)
    expected_restarts = len(ids_res["path_actions"]) + 1  # depth limits 0 .. d
    report.check(
        ids_starts == expected_restarts,
        f"IDS signature: restarted from root {ids_starts} times (expected {expected_restarts} for depth 0..{expected_restarts - 1})",
    )
    report.check(
        ids_res["max_frontier_size"] < bfs_res["max_frontier_size"]
        and ids_res["nodes_expanded"] > bfs_res["nodes_expanded"],
        f"IDS signature: smaller frontier ({ids_res['max_frontier_size']} < {bfs_res['max_frontier_size']}) "
        f"and higher re-expansions ({ids_res['nodes_expanded']} > {bfs_res['nodes_expanded']}) than BFS",
    )

    # 4. DFS Signature: first step follows LIFO stack priority ('UP' -> 'DOWN' -> 'LEFT' -> 'RIGHT')
    dfs_order = []
    dfs(problem, trace=lambda s, exp, front: dfs_order.append(s) if len(dfs_order) < 2 else None)
    first_expected_successor = problem.get_successors(start)[0][1]
    report.check(
        len(dfs_order) >= 2 and dfs_order[1] == first_expected_successor,
        f"DFS signature: immediately dives into first action successor {first_expected_successor} (LIFO stack)",
    )
def main():
    report = Report()
    print("CSAI 301 - PHASE I SEARCH VALIDATION")
    print("Checks problem model, path legality, costs, metrics, and optimality.")

    validate_default_map(report)
    validate_weighted_map(report)
    validate_algorithm_signatures(report)

    report.summary()
    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())
