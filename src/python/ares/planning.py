from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    obstacles: frozenset[tuple[int, int]]

    def in_bounds(self, node: tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, node: tuple[int, int]) -> bool:
        return node not in self.obstacles

    def neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in candidates if self.in_bounds(n) and self.passable(n)]

    def with_obstacles(self, new_obstacles: set[tuple[int, int]]) -> "GridMap":
        return GridMap(self.width, self.height, frozenset(new_obstacles))


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    node = goal
    path = [node]
    while came_from[node] is not None:
        node = came_from[node]  # type: ignore[assignment]
        path.append(node)
    path.reverse()
    return path


def dijkstra(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    came_from = {start: None}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct_path(came_from, goal)
        if current_cost > cost[current]:
            continue
        for neighbor in grid.neighbors(current):
            new_cost = current_cost + 1.0
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))
    raise ValueError("No path found")


def astar(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    frontier = [(0.0, start)]
    g_score = {start: 0.0}
    came_from = {start: None}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct_path(came_from, goal)
        for neighbor in grid.neighbors(current):
            tentative = g_score[current] + 1.0
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_score, neighbor))
    raise ValueError("No path found")


def rrt(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    iterations: int = 1000,
    seed: int = 7,
) -> list[tuple[int, int]] | None:
    rng = random.Random(seed)
    parents = {start: None}
    nodes = [start]
    for _ in range(iterations):
        sample = goal if rng.random() < 0.15 else (
            rng.randrange(grid.width),
            rng.randrange(grid.height),
        )
        nearest = min(nodes, key=lambda node: abs(node[0] - sample[0]) + abs(node[1] - sample[1]))
        dx = 0 if sample[0] == nearest[0] else (1 if sample[0] > nearest[0] else -1)
        dy = 0 if sample[1] == nearest[1] else (1 if sample[1] > nearest[1] else -1)
        candidate = (nearest[0] + dx, nearest[1] + dy)
        if not grid.in_bounds(candidate) or not grid.passable(candidate) or candidate in parents:
            continue
        parents[candidate] = nearest
        nodes.append(candidate)
        if candidate == goal or abs(candidate[0] - goal[0]) + abs(candidate[1] - goal[1]) <= 1:
            parents[goal] = candidate if candidate != goal else nearest
            return _reconstruct_path(parents, goal)
    return None


def rrt_star(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    iterations: int = 1500,
    rewire_radius: float = 2.5,
    seed: int = 19,
) -> list[tuple[int, int]] | None:
    rng = random.Random(seed)
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    costs: dict[tuple[int, int], float] = {start: 0.0}
    nodes = [start]
    for _ in range(iterations):
        sample = goal if rng.random() < 0.2 else (rng.randrange(grid.width), rng.randrange(grid.height))
        nearest = min(nodes, key=lambda node: abs(node[0] - sample[0]) + abs(node[1] - sample[1]))
        dx = 0 if sample[0] == nearest[0] else (1 if sample[0] > nearest[0] else -1)
        dy = 0 if sample[1] == nearest[1] else (1 if sample[1] > nearest[1] else -1)
        candidate = (nearest[0] + dx, nearest[1] + dy)
        if not grid.in_bounds(candidate) or not grid.passable(candidate):
            continue
        neighbors = [node for node in nodes if math.dist(node, candidate) <= rewire_radius]
        parent = nearest
        parent_cost = costs[nearest] + math.dist(nearest, candidate)
        for neighbor in neighbors:
            candidate_cost = costs[neighbor] + math.dist(neighbor, candidate)
            if candidate_cost < parent_cost:
                parent = neighbor
                parent_cost = candidate_cost
        if candidate not in costs or parent_cost < costs[candidate]:
            parents[candidate] = parent
            costs[candidate] = parent_cost
            if candidate not in nodes:
                nodes.append(candidate)
            for neighbor in neighbors:
                rewired = costs[candidate] + math.dist(candidate, neighbor)
                if rewired < costs[neighbor]:
                    parents[neighbor] = candidate
                    costs[neighbor] = rewired
        if math.dist(candidate, goal) <= 1.0:
            parents[goal] = candidate
            return _reconstruct_path(parents, goal)
    return None


def prm(
    grid: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    samples: int = 120,
    k_neighbors: int = 8,
    seed: int = 23,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    nodes = {start, goal}
    while len(nodes) < samples + 2:
        candidate = (rng.randrange(grid.width), rng.randrange(grid.height))
        if grid.passable(candidate):
            nodes.add(candidate)
    node_list = list(nodes)
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {node: [] for node in node_list}

    def collision_free(a: tuple[int, int], b: tuple[int, int]) -> bool:
        steps = max(abs(a[0] - b[0]), abs(a[1] - b[1]), 1)
        for step in range(steps + 1):
            t = step / steps
            sample = (round(a[0] + t * (b[0] - a[0])), round(a[1] + t * (b[1] - a[1])))
            if not grid.in_bounds(sample) or not grid.passable(sample):
                return False
        return True

    for node in node_list:
        ordered = sorted((other for other in node_list if other != node), key=lambda other: math.dist(node, other))
        for other in ordered[:k_neighbors]:
            if collision_free(node, other):
                cost = math.dist(node, other)
                graph[node].append((other, cost))
                graph[other].append((node, cost))

    frontier = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct_path(came_from, goal)
        if current_cost > cost_so_far[current]:
            continue
        for neighbor, edge_cost in graph[current]:
            new_cost = current_cost + edge_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))
    raise ValueError("No PRM path found")


class DStarLitePlanner:
    """A compact replanning wrapper that reuses A* after local map updates."""

    def __init__(self, grid: GridMap) -> None:
        self.grid = grid

    def update_obstacles(self, obstacles: set[tuple[int, int]]) -> None:
        self.grid = self.grid.with_obstacles(obstacles)

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        return astar(self.grid, start, goal)


def path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(path)):
        total += math.dist(path[idx - 1], path[idx])
    return total


def dubins_like_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    turning_radius: float = 1.0,
) -> list[tuple[float, float]]:
    midpoint = ((start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0 + turning_radius)
    return [start, midpoint, goal]


def reeds_shepp_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    turning_radius: float = 1.0,
) -> list[tuple[float, float]]:
    midpoint = ((start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0 - turning_radius)
    reverse_point = (midpoint[0] - 0.5 * turning_radius, midpoint[1])
    return [start, midpoint, reverse_point, goal]


def clothoid_curve(
    start: tuple[float, float],
    goal: tuple[float, float],
    samples: int = 20,
) -> list[tuple[float, float]]:
    points = []
    for idx in range(samples):
        t = idx / max(samples - 1, 1)
        blend = 3.0 * t ** 2 - 2.0 * t ** 3
        x = start[0] + (goal[0] - start[0]) * t
        y = start[1] + (goal[1] - start[1]) * blend
        points.append((x, y))
    return points


def minimum_snap_trajectory(
    waypoints: list[tuple[float, float]],
    samples_per_segment: int = 10,
) -> list[tuple[float, float]]:
    trajectory: list[tuple[float, float]] = []
    for idx in range(1, len(waypoints)):
        start = waypoints[idx - 1]
        goal = waypoints[idx]
        for step in range(samples_per_segment):
            t = step / max(samples_per_segment - 1, 1)
            s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
            trajectory.append(
                (
                    start[0] + (goal[0] - start[0]) * s,
                    start[1] + (goal[1] - start[1]) * s,
                )
            )
    return trajectory


def chomp_smooth_path(
    path: list[tuple[float, float]],
    obstacles: list[tuple[float, float]],
    iterations: int = 40,
    smooth_weight: float = 0.2,
    obstacle_weight: float = 0.08,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    points = [np.array(point, dtype=float) for point in path]
    for _ in range(iterations):
        for idx in range(1, len(points) - 1):
            smooth_grad = 2.0 * points[idx] - points[idx - 1] - points[idx + 1]
            obstacle_grad = np.zeros(2, dtype=float)
            for obstacle in obstacles:
                delta = points[idx] - np.asarray(obstacle, dtype=float)
                dist = np.linalg.norm(delta)
                if dist < 1.0 and dist > 1e-9:
                    obstacle_grad += delta / (dist ** 3)
            points[idx] -= smooth_weight * smooth_grad - obstacle_weight * obstacle_grad
    return [tuple(point.tolist()) for point in points]


def trajopt_optimize(
    path: list[tuple[float, float]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    iterations: int = 30,
) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    x_bounds, y_bounds = bounds
    points = [np.array(point, dtype=float) for point in path]
    for _ in range(iterations):
        for idx in range(1, len(points) - 1):
            target = 0.5 * (points[idx - 1] + points[idx + 1])
            points[idx] = 0.7 * points[idx] + 0.3 * target
            points[idx][0] = np.clip(points[idx][0], x_bounds[0], x_bounds[1])
            points[idx][1] = np.clip(points[idx][1], y_bounds[0], y_bounds[1])
    return [tuple(point.tolist()) for point in points]
