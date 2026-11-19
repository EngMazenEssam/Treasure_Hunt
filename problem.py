"""
problem.py
Formal problem formulation for the Treasure Hunt Grid World (CSAI 301 - Phase I).

State Representation:
    s = (row, col, treasure_mask, has_key)
    - row, col: integer coordinates on the 15x15 grid
    - treasure_mask: 4-bit integer (0..15) tracking collected treasures T0..T3
    - has_key: binary integer (0 or 1) indicating whether Key 'K' is collected
"""


class TreasureHuntProblem:
    """Encapsulates the state space, transition model, step costs, and goal test."""

    DEFAULT_MAP = [
        "P..#...SS....0.",
        ".#.#.#.SM.###..",
        ".#...#....#1...",
        ".###.####.#.##.",
        "............SS.",
        "##.###.####.SM.",
        "K....#....#....",
        "####.#.##.###.#",
        ".......#.......",
        ".#####.#.#####.",
        "...2.#....SS...",
        "####.####.SM.##",
        ".............DE",
        ".######.#######",
        "............3..",
    ]

    def __init__(self, grid_lines=None):
        self.grid = [list(row) for row in (grid_lines or self.DEFAULT_MAP)]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # Terrain step costs: Grass=1, Sand=3, Mud=5
        self.step_costs = {
            ".": 1,
            "P": 1,
            "E": 1,
            "K": 1,
            "D": 1,
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 1,
            "S": 3,
            "M": 5,
        }

        self.start_pos = None
        self.exit_pos = None
        self.key_pos = None
        self.door_pos = None
        self.treasures = {}  # (r, c) -> bit index (0..3)

        self._parse_entities()
        self.all_treasures_mask = (1 << len(self.treasures)) - 1

    def _parse_entities(self):
        """Scans the grid to locate Start, Exit, Key, Door, and Treasures."""
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell == "P":
                    self.start_pos = (r, c)
                elif cell == "E":
                    self.exit_pos = (r, c)
                elif cell == "K":
                    self.key_pos = (r, c)
                elif cell == "D":
                    self.door_pos = (r, c)
                elif cell in {"0", "1", "2", "3"}:
                    self.treasures[(r, c)] = int(cell)

    def get_initial_state(self):
        """Returns initial state s0 = (r_start, c_start, 0, 0)."""
        r, c = self.start_pos
        t_mask = 1 << self.treasures[(r, c)] if (r, c) in self.treasures else 0
        has_key = 1 if (r, c) == self.key_pos else 0
        return (r, c, t_mask, has_key)

    def is_goal(self, state):
        """
        Goal test: agent is at Exit 'E', has collected all 4 treasures, and has
        the key. The key condition is redundant on this map (the exit is behind
        the locked door) but makes the goal definition explicit and robust.
        """
        r, c, t_mask, has_key = state
        return (
            (r, c) == self.exit_pos
            and t_mask == self.all_treasures_mask
            and has_key == 1
        )

    def get_successors(self, state):
        """
        Generates valid transitions in fixed deterministic order:
        UP, DOWN, LEFT, RIGHT.

        Returns:
            list of tuples: (action_name, next_state, step_cost)
        """
        r, c, t_mask, has_key = state
        actions = [
            ("UP", -1, 0),
            ("DOWN", 1, 0),
            ("LEFT", 0, -1),
            ("RIGHT", 0, 1),
        ]
        successors = []

        for action, dr, dc in actions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue

            cell = self.grid[nr][nc]
            # Walls are always impassable
            if cell == "#":
                continue
            # Locked door requires the key
            if cell == "D" and has_key == 0:
                continue

            new_t_mask = t_mask
            if (nr, nc) in self.treasures:
                new_t_mask |= 1 << self.treasures[(nr, nc)]

            new_has_key = 1 if (has_key == 1 or (nr, nc) == self.key_pos) else 0
            cost = self.step_costs[cell]

            next_state = (nr, nc, new_t_mask, new_has_key)
            successors.append((action, next_state, cost))

        return successors

    def count_reachable_states(self):
        """
        Counts the exact number of states reachable from the initial state.

        The theoretical product (walkable cells x treasure masks x key flag) is only
        an upper bound: many combinations are impossible (e.g. has_key=0 beyond the
        locked door, or treasure masks that require walking through the door first).
        A graph traversal over the implicit state space gives the exact size.
        """
        from collections import deque

        start = self.get_initial_state()
        seen = {start}
        queue = deque([start])

        while queue:
            state = queue.popleft()
            for _, next_state, _ in self.get_successors(state):
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append(next_state)

        return len(seen)

    def get_state_space_stats(self):
        """Computes grid statistics, theoretical upper bound, and exact reachable size."""
        total_cells = self.rows * self.cols
        wall_cells = sum(row.count("#") for row in self.grid)
        walkable_cells = total_cells - wall_cells
        num_treasure_subsets = 1 << len(self.treasures)  # 2^4 = 16
        num_key_states = 2  # has_key in {0, 1}
        theoretical_states = walkable_cells * num_treasure_subsets * num_key_states
        reachable_states = self.count_reachable_states()

        return {
            "rows": self.rows,
            "cols": self.cols,
            "total_cells": total_cells,
            "wall_cells": wall_cells,
            "walkable_cells": walkable_cells,
            "treasure_count": len(self.treasures),
            "theoretical_states": theoretical_states,
            "reachable_states": reachable_states,
        }