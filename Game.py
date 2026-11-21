import pygame
import random
import time
import heapq
from collections import deque

# =========================
# Initialize Pygame
# =========================
pygame.init()

WIDTH, HEIGHT = 1000, 750
CELL_SIZE = 35
ROWS, COLS = 15, 15

GRID_X = 40
GRID_Y = 120

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treasure Hunt - Search Visualizer")

clock = pygame.time.Clock()

# =========================
# Colors
# =========================
BG = (18, 23, 38)
PANEL = (30, 38, 58)
WHITE = (240, 240, 240)
GREEN = (40, 200, 120)
BLUE = (60, 140, 255)
RED = (255, 80, 80)
YELLOW = (255, 215, 0)
GRAY = (100, 100, 100)
BLACK = (10, 10, 10)

# =========================
# Fonts
# =========================
font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 17)
big_font = pygame.font.SysFont("Arial", 32, bold=True)

# =========================
# Maze
# 0 = Empty
# 1 = Wall
# =========================
maze = [
    [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
    [0,1,0,1,0,1,1,0,1,0,1,1,1,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [0,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,0,1,0,1,1,1,0,1],
    [0,0,0,0,0,0,1,0,1,0,0,0,1,0,0],
    [0,1,1,1,1,0,1,0,1,1,1,0,1,1,0],
    [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
    [0,1,1,0,1,1,1,1,1,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
    [1,1,1,1,1,0,1,0,1,1,1,1,0,1,0],
    [0,0,0,0,1,0,1,0,0,0,0,1,0,0,0],
    [0,1,1,0,1,0,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

START = (0, 0)
GOAL = (14, 14)

# =========================
# Game State
# =========================
algorithm = "BFS"

explored = []
path = []

running_search = False
paused = False
search_finished = False

nodes_expanded = 0
nodes_generated = 0
max_frontier = 0
path_cost = 0
execution_time = 0

speed = 80
search_generator = None

# =========================
# Buttons
# =========================
buttons = {
    "BFS": pygame.Rect(600, 120, 85, 40),
    "DFS": pygame.Rect(695, 120, 85, 40),
    "UCS": pygame.Rect(790, 120, 85, 40),
    "IDS": pygame.Rect(885, 120, 85, 40),

    "PLAY": pygame.Rect(600, 185, 120, 45),
    "PAUSE": pygame.Rect(730, 185, 120, 45),
    "RESET": pygame.Rect(860, 185, 120, 45),
}

# =========================
# Helper Functions
# =========================
def neighbors(state):
    r, c = state

    moves = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    result = []

    for dr, dc in moves:
        nr, nc = r + dr, c + dc

        if (
            0 <= nr < ROWS
            and 0 <= nc < COLS
            and maze[nr][nc] == 0
        ):
            result.append((nr, nc))

    return result


def reconstruct_path(parent, current):
    result = [current]

    while current in parent:
        current = parent[current]
        result.append(current)

    result.reverse()
    return result


def draw_text(text, x, y, color=WHITE, f=font):
    surface = f.render(str(text), True, color)
    screen.blit(surface, (x, y))


# =========================
# Search Algorithms
# =========================
def bfs():
    global nodes_expanded, nodes_generated, max_frontier

    queue = deque([START])
    parent = {}
    visited = {START}

    nodes_generated = 1

    while queue:
        max_frontier = max(max_frontier, len(queue))

        current = queue.popleft()

        nodes_expanded += 1
        explored.append(current)

        yield current

        if current == GOAL:
            return reconstruct_path(parent, current)

        for nxt in neighbors(current):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)
                nodes_generated += 1

    return []


def dfs():
    global nodes_expanded, nodes_generated, max_frontier

    stack = [START]
    parent = {}
    visited = {START}

    nodes_generated = 1

    while stack:
        max_frontier = max(max_frontier, len(stack))

        current = stack.pop()

        nodes_expanded += 1
        explored.append(current)

        yield current

        if current == GOAL:
            return reconstruct_path(parent, current)

        for nxt in reversed(neighbors(current)):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                stack.append(nxt)
                nodes_generated += 1

    return []


def ucs():
    global nodes_expanded, nodes_generated, max_frontier

    pq = [(0, START)]
    parent = {}
    cost_so_far = {START: 0}

    nodes_generated = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))

        cost, current = heapq.heappop(pq)

        if cost != cost_so_far[current]:
            continue

        nodes_expanded += 1
        explored.append(current)

        yield current

        if current == GOAL:
            return reconstruct_path(parent, current)

        for nxt in neighbors(current):
            new_cost = cost + 1

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                parent[nxt] = current

                heapq.heappush(pq, (new_cost, nxt))
                nodes_generated += 1

    return []


def ids():
    global nodes_expanded, nodes_generated, max_frontier

    depth_limit = 0

    while True:
        stack = [(START, 0)]
        parent = {}
        visited_depth = {START: 0}

        while stack:
            max_frontier = max(max_frontier, len(stack))

            current, depth = stack.pop()

            nodes_expanded += 1
            explored.append(current)

            yield current

            if current == GOAL:
                return reconstruct_path(parent, current)

            if depth < depth_limit:
                for nxt in reversed(neighbors(current)):
                    if nxt not in visited_depth or depth + 1 < visited_depth[nxt]:
                        visited_depth[nxt] = depth + 1
                        parent[nxt] = current
                        stack.append((nxt, depth + 1))
                        nodes_generated += 1

        depth_limit += 1


# =========================
# Start Search
# =========================
def start_search():
    global running_search, paused, search_finished
    global nodes_expanded, nodes_generated, max_frontier
    global path, path_cost, execution_time
    global explored, search_generator

    explored = []
    path = []

    nodes_expanded = 0
    nodes_generated = 0
    max_frontier = 0
    path_cost = 0
    execution_time = 0

    search_finished = False
    paused = False
    running_search = True

    if algorithm == "BFS":
        search_generator = bfs()
    elif algorithm == "DFS":
        search_generator = dfs()
    elif algorithm == "UCS":
        search_generator = ucs()
    elif algorithm == "IDS":
        search_generator = ids()

    search_start_time = time.perf_counter()


# =========================
# Drawing Maze
# =========================
def draw_maze():
    for r in range(ROWS):
        for c in range(COLS):

            rect = pygame.Rect(
                GRID_X + c * CELL_SIZE,
                GRID_Y + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            if maze[r][c] == 1:
                color = (40, 45, 65)

            elif (r, c) in path:
                color = YELLOW

            elif (r, c) in explored:
                color = BLUE

            else:
                color = (225, 230, 235)

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Start
    start_rect = pygame.Rect(
        GRID_X + START[1] * CELL_SIZE,
        GRID_Y + START[0] * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
    )

    pygame.draw.rect(screen, GREEN, start_rect)
    draw_text("S", start_rect.x + 9, start_rect.y + 5, BLACK)

    # Goal
    goal_rect = pygame.Rect(
        GRID_X + GOAL[1] * CELL_SIZE,
        GRID_Y + GOAL[0] * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE
    )

    pygame.draw.rect(screen, YELLOW, goal_rect)
    draw_text("T", goal_rect.x + 9, goal_rect.y + 5, BLACK)


# =========================
# Draw Buttons
# =========================
def draw_buttons():
    for name, rect in buttons.items():

        if name == algorithm:
            color = (50, 150, 220)

        elif name == "PLAY":
            color = GREEN

        elif name == "PAUSE":
            color = (230, 160, 50)

        elif name == "RESET":
            color = RED

        else:
            color = (60, 75, 100)

        pygame.draw.rect(screen, color, rect, border_radius=8)

        label = font.render(name, True, WHITE)

        screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2
            )
        )


# =========================
# Draw Statistics
# =========================
def draw_stats():
    x = 600
    y = 270

    draw_text("SEARCH STATISTICS", x, y, YELLOW, big_font)
    y += 50

    draw_text(f"Algorithm: {algorithm}", x, y)
    y += 35

    draw_text(f"Expanded: {nodes_expanded}", x, y)
    y += 35

    draw_text(f"Generated: {nodes_generated}", x, y)
    y += 35

    draw_text(f"Frontier: {max_frontier}", x, y)
    y += 35

    draw_text(f"Path Cost: {path_cost}", x, y)
    y += 35

    draw_text(f"Explored Cells: {len(explored)}", x, y)
    y += 35

    draw_text(f"Speed: {speed}", x, y)
    y += 35

    if search_finished:
        draw_text("TREASURE FOUND!", x, y, GREEN, big_font)


# =========================
# Main Loop
# =========================
while True:

    screen.fill(BG)

    draw_text("TREASURE HUNT", 40, 30, YELLOW, big_font)

    draw_maze()
    draw_buttons()
    draw_stats()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.MOUSEBUTTONDOWN:

            pos = event.pos

            # Select Algorithm
            for name in ["BFS", "DFS", "UCS", "IDS"]:
                if buttons[name].collidepoint(pos):
                    algorithm = name
                    running_search = False
                    search_finished = False
                    explored = []
                    path = []

            # Play
            if buttons["PLAY"].collidepoint(pos):
                start_search()

            # Pause
            if buttons["PAUSE"].collidepoint(pos):
                paused = not paused

            # Reset
            if buttons["RESET"].collidepoint(pos):
                running_search = False
                paused = False
                search_finished = False
                explored = []
                path = []
                nodes_expanded = 0
                nodes_generated = 0
                max_frontier = 0
                path_cost = 0

    # Animate Search
    if running_search and not paused and not search_finished:

        try:
            next(search_generator)

        except StopIteration as e:
            path = e.value if e.value else []

            path_cost = max(0, len(path) - 1)

            search_finished = True
            running_search = False

    pygame.display.flip()
    clock.tick(12)

pygame.quit()