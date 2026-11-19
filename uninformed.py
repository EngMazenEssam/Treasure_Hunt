"""
uninformed.py
From-scratch implementations of Uninformed Search algorithms (CSAI 301 - Phase I):
    1. Breadth-First Search (BFS)
    2. Depth-First Search (DFS)
    3. Uniform-Cost Search (UCS)
    4. Iterative Deepening Search (IDS)
"""

from collections import deque
import heapq
import time


def _build_result(
    algorithm,
    found,
    path_actions,
    path_states,
    path_cost,
    nodes_generated,
    nodes_expanded,
    max_frontier_size,
    start_time,
    visited_coords,
    
):
    """Standardized return dictionary across all search algorithms."""
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "algorithm": algorithm,
        "found": found,
        "path_actions": path_actions,
        "path_states": path_states,
        "path_cost": path_cost,
        "nodes_generated": nodes_generated,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier_size,
        "execution_time_ms": elapsed_ms,
        "visited_coords": visited_coords,
    }


def bfs(problem, trace=None):
    """
    Breadth-First Search (Graph Search).
    - Uses a FIFO queue (deque) and an explored state set.
    - Complete and optimal for unweighted step count, NOT weighted path cost.
    - Time & Space Complexity: O(b^d) where b <= 4 and d is solution depth.

    Args:
        problem: TreasureHuntProblem instance.
        trace: optional callback ``trace(state, expanded_count, frontier_size)``
               invoked once per expanded node (used by animate.py).
    """
    start_time = time.perf_counter()
    initial_state = problem.get_initial_state()

    frontier = deque([(initial_state, [], [initial_state], 0)])
    explored = {initial_state}
    visited_coords = {}

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state, actions, states, cost = frontier.popleft()
        nodes_expanded += 1

        coord = (state[0], state[1])
        visited_coords[coord] = visited_coords.get(coord, 0) + 1
        if trace is not None:
            trace(state, nodes_expanded, len(frontier))

        if problem.is_goal(state):
            return _build_result(
                "BFS",
                True,
                actions,
                states,
                cost,
                nodes_generated,
                nodes_expanded,
                max_frontier_size,
                start_time,
                visited_coords,
            )

        for action, next_state, step_cost in problem.get_successors(state):
            nodes_generated += 1
            if next_state not in explored:
                explored.add(next_state)
                frontier.append(
                    (next_state, actions + [action], states + [next_state], cost + step_cost)
                )

    return _build_result(
        "BFS", False, [], [], 0, nodes_generated, nodes_expanded, max_frontier_size, start_time, visited_coords
    )


def dfs(problem, trace=None):
    """
    Depth-First Search (Graph Search).
    - Uses a LIFO stack and an explored state set to prevent infinite grid loops.
    - Pushes successors in reverse order so 'UP' is expanded first.
    - Complete on finite state spaces, NOT optimal.
    - Time Complexity: O(|S|), Space Complexity: O(|S|) with explored set.

    Args:
        problem: TreasureHuntProblem instance.
        trace: optional callback ``trace(state, expanded_count, frontier_size)``
               invoked once per expanded node (used by animate.py).
    """
    start_time = time.perf_counter()
    initial_state = problem.get_initial_state()

    frontier = [(initial_state, [], [initial_state], 0)]
    explored = set()
    visited_coords = {}

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        state, actions, states, cost = frontier.pop()

        if state in explored:
            continue
        explored.add(state)
        nodes_expanded += 1

        coord = (state[0], state[1])
        visited_coords[coord] = visited_coords.get(coord, 0) + 1
        if trace is not None:
            trace(state, nodes_expanded, len(frontier))

        if problem.is_goal(state):
            return _build_result(
                "DFS",
                True,
                actions,
                states,
                cost,
                nodes_generated,
                nodes_expanded,
                max_frontier_size,
                start_time,
                visited_coords,
            )

        successors = problem.get_successors(state)
        # Reverse so first action ('UP') is at top of stack
        for action, next_state, step_cost in reversed(successors):
            nodes_generated += 1
            if next_state not in explored:
                frontier.append(
                    (next_state, actions + [action], states + [next_state], cost + step_cost)
                )

    return _build_result(
        "DFS", False, [], [], 0, nodes_generated, nodes_expanded, max_frontier_size, start_time, visited_coords
    )


def ucs(problem, trace=None):
    """
    Uniform-Cost Search (Dijkstra's Algorithm on implicit state graph).
    - Uses a min-heap ordered by cumulative path cost g(n).
    - Complete and guaranteed OPTIMAL for positive step costs (>= 1).
    - Time & Space Complexity: O(b^(1 + floor(C* / eps))).

    Args:
        problem: TreasureHuntProblem instance.
        trace: optional callback ``trace(state, expanded_count, frontier_size)``
               invoked once per expanded node (used by animate.py).
    """
    start_time = time.perf_counter()
    initial_state = problem.get_initial_state()

    counter = 0
    frontier = [(0, counter, initial_state, [], [initial_state])]
    best_cost = {initial_state: 0}
    visited_coords = {}

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        cost, _, state, actions, states = heapq.heappop(frontier)

        # Skip stale heap entries
        if cost > best_cost.get(state, float("inf")):
            continue

        nodes_expanded += 1
        coord = (state[0], state[1])
        visited_coords[coord] = visited_coords.get(coord, 0) + 1
        if trace is not None:
            trace(state, nodes_expanded, len(frontier))

        if problem.is_goal(state):
            return _build_result(
                "UCS",
                True,
                actions,
                states,
                cost,
                nodes_generated,
                nodes_expanded,
                max_frontier_size,
                start_time,
                visited_coords,
            )

        for action, next_state, step_cost in problem.get_successors(state):
            nodes_generated += 1
            new_cost = cost + step_cost
            if new_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = new_cost
                counter += 1
                heapq.heappush(
                    frontier,
                    (new_cost, counter, next_state, actions + [action], states + [next_state]),
                )

    return _build_result(
        "UCS", False, [], [], 0, nodes_generated, nodes_expanded, max_frontier_size, start_time, visited_coords
    )


def ids(problem, max_depth=250, trace=None):
    """
    Iterative Deepening Search (IDS).
    - Repeatedly executes Depth-Limited Search (DLS) with increasing depth limits.
    - Uses per-iteration depth pruning (`min_depth_in_iter[state] <= depth`) to avoid
      exponential grid cycles while preserving minimum-step optimality.
    - Time Complexity: O(b^d), Space Complexity: O(b*d) frontier memory.

    Args:
        problem: TreasureHuntProblem instance.
        max_depth: safety cap on the depth limit.
        trace: optional callback ``trace(state, expanded_count, frontier_size)``
               invoked once per expanded node (used by animate.py).
    """
    start_time = time.perf_counter()
    initial_state = problem.get_initial_state()

    total_nodes_generated = 0
    total_nodes_expanded = 0
    overall_max_frontier = 1
    visited_coords = {}

    for depth_limit in range(max_depth + 1):
        # Iterative stack for DLS: (state, actions, states, cost, depth)
        stack = [(initial_state, [], [initial_state], 0, 0)]
        min_depth_in_iter = {initial_state: 0}
        total_nodes_generated += 1

        while stack:
            overall_max_frontier = max(overall_max_frontier, len(stack))
            state, actions, states, cost, depth = stack.pop()
            total_nodes_expanded += 1

            coord = (state[0], state[1])
            visited_coords[coord] = visited_coords.get(coord, 0) + 1
            if trace is not None:
                trace(state, total_nodes_expanded, len(stack))

            if problem.is_goal(state):
                return _build_result(
                    "IDS",
                    True,
                    actions,
                    states,
                    cost,
                    total_nodes_generated,
                    total_nodes_expanded,
                    overall_max_frontier,
                    start_time,
                    visited_coords,
                )

            if depth < depth_limit:
                successors = problem.get_successors(state)
                for action, next_state, step_cost in reversed(successors):
                    total_nodes_generated += 1
                    next_depth = depth + 1
                    if next_depth < min_depth_in_iter.get(next_state, float("inf")):
                        min_depth_in_iter[next_state] = next_depth
                        stack.append(
                            (
                                next_state,
                                actions + [action],
                                states + [next_state],
                                cost + step_cost,
                                next_depth,
                            )
                        )

    return _build_result(
        "IDS",
        False,
        [],
        [],
        0,
        total_nodes_generated,
        total_nodes_expanded,
        overall_max_frontier,
        start_time,
        visited_coords,
    )