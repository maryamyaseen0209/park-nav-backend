from __future__ import annotations

import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .data import GRAPH, POSITIONS


@dataclass
class SearchOutcome:
    found: bool
    path: list[str] = field(default_factory=list)
    total_time: int | None = None
    visited_order: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ParkSearchEngine:
    def __init__(self, graph: dict[str, dict[str, int]] | None = None):
        self.graph = graph or GRAPH

    def validate_nodes(self, start: str, goal: str) -> None:
        if start not in self.graph:
            raise ValueError(f"Unknown start park: {start}")
        if goal not in self.graph:
            raise ValueError(f"Unknown destination park: {goal}")
        if start == goal:
            raise ValueError("Start and destination must be different parks")

    def path_time(self, path: list[str]) -> int:
        total = 0
        for current, nxt in zip(path, path[1:]):
            try:
                total += self.graph[current][nxt]
            except KeyError as exc:
                raise ValueError(f"Invalid route segment: {current} -> {nxt}") from exc
        return total

    def _within_constraints(self, path: list[str], time_limit: int, max_stops: int) -> bool:
        return len(path) - 1 <= max_stops and self.path_time(path) <= time_limit

    def bfs(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        queue: deque[tuple[list[str], int]] = deque([([start], 0)])
        # Time is part of the state because a cheaper path to the same node at the
        # same depth may remain feasible under a travel-time constraint.
        best_time_by_state: dict[tuple[str, int], int] = {(start, 0): 0}
        visited_order: list[str] = []
        trace: list[dict[str, Any]] = []

        while queue:
            path, total = queue.popleft()
            node = path[-1]
            depth = len(path) - 1
            state = (node, depth)
            if total != best_time_by_state.get(state):
                continue

            visited_order.append(node)
            trace.append(
                {
                    "action": "expand",
                    "node": node,
                    "frontier": len(queue),
                    "depth": depth,
                    "time": total,
                    "path": path,
                }
            )

            if node == goal:
                return SearchOutcome(True, path, total, visited_order, trace)

            if depth >= max_stops:
                continue

            for neighbor, edge_cost in self.graph[node].items():
                if neighbor in path:
                    continue
                candidate_total = total + edge_cost
                if candidate_total > time_limit:
                    continue
                next_depth = depth + 1
                next_state = (neighbor, next_depth)
                if candidate_total < best_time_by_state.get(next_state, math.inf):
                    best_time_by_state[next_state] = candidate_total
                    queue.append((path + [neighbor], candidate_total))
                    trace.append(
                        {
                            "action": "enqueue",
                            "node": neighbor,
                            "from": node,
                            "depth": next_depth,
                            "time": candidate_total,
                        }
                    )

        return SearchOutcome(False, visited_order=visited_order, trace=trace, warning="No valid BFS route found.")

    def dfs(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        stack: list[tuple[list[str], int]] = [([start], 0)]
        visited_order: list[str] = []
        trace: list[dict[str, Any]] = []

        while stack:
            path, total = stack.pop()
            node = path[-1]
            visited_order.append(node)
            trace.append(
                {"action": "expand", "node": node, "stack": len(stack), "time": total, "path": path}
            )

            if node == goal:
                return SearchOutcome(True, path, total, visited_order, trace)

            if len(path) - 1 >= max_stops:
                continue

            # Reverse to make the result deterministic while preserving notebook order.
            for neighbor, edge_cost in reversed(list(self.graph[node].items())):
                if neighbor in path:
                    continue
                candidate_total = total + edge_cost
                if candidate_total <= time_limit:
                    stack.append((path + [neighbor], candidate_total))
                    trace.append(
                        {"action": "push", "node": neighbor, "from": node, "time": candidate_total}
                    )

        return SearchOutcome(False, visited_order=visited_order, trace=trace, warning="DFS exhausted valid branches.")

    def ucs(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        frontier: list[tuple[int, int, str, list[str]]] = [(0, 0, start, [start])]
        best_cost: dict[tuple[str, int], int] = {(start, 0): 0}
        visited_order: list[str] = []
        trace: list[dict[str, Any]] = []
        counter = 1

        while frontier:
            cost, _, node, path = heapq.heappop(frontier)
            depth = len(path) - 1
            state = (node, depth)
            if cost != best_cost.get(state):
                continue
            visited_order.append(node)
            trace.append(
                {"action": "expand", "node": node, "cost": cost, "depth": depth, "frontier": len(frontier)}
            )

            if node == goal:
                return SearchOutcome(True, path, cost, visited_order, trace)

            if depth >= max_stops:
                continue

            for neighbor, edge_cost in self.graph[node].items():
                if neighbor in path:
                    continue
                new_cost = cost + edge_cost
                if new_cost > time_limit:
                    continue
                next_state = (neighbor, depth + 1)
                if new_cost < best_cost.get(next_state, math.inf):
                    best_cost[next_state] = new_cost
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [neighbor]))
                    counter += 1
                    trace.append(
                        {"action": "relax", "from": node, "node": neighbor, "cost": new_cost, "depth": depth + 1}
                    )

        return SearchOutcome(False, visited_order=visited_order, trace=trace, warning="No route fits the travel-time limit.")

    def iddfs(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        complete_visit_log: list[str] = []
        trace: list[dict[str, Any]] = []

        def depth_limited(path: list[str], depth_left: int) -> list[str] | None:
            node = path[-1]
            complete_visit_log.append(node)
            trace.append({"action": "visit", "node": node, "depth_left": depth_left, "path": path})
            if node == goal:
                return path
            if depth_left == 0:
                return None

            for neighbor in self.graph[node]:
                if neighbor in path:
                    continue
                candidate = path + [neighbor]
                if self.path_time(candidate) > time_limit:
                    continue
                result = depth_limited(candidate, depth_left - 1)
                if result:
                    return result
            return None

        for depth in range(max_stops + 1):
            trace.append({"action": "new_depth", "depth": depth})
            result = depth_limited([start], depth)
            if result:
                return SearchOutcome(
                    True,
                    result,
                    self.path_time(result),
                    complete_visit_log,
                    trace,
                    metadata={"depth_found": depth},
                )

        return SearchOutcome(
            False,
            visited_order=complete_visit_log,
            trace=trace,
            warning="No route was found within the iterative depth and time limits.",
        )

    def _heuristic_scale(self) -> float:
        ratios: list[float] = []
        seen: set[tuple[str, str]] = set()
        for node, neighbors in self.graph.items():
            for neighbor, cost in neighbors.items():
                key = tuple(sorted((node, neighbor)))
                if key in seen:
                    continue
                seen.add(key)
                distance = self._distance(node, neighbor)
                if distance > 0:
                    ratios.append(cost / distance)
        return min(ratios) if ratios else 0.0

    def _distance(self, first: str, second: str) -> float:
        x1, y1 = POSITIONS[first]
        x2, y2 = POSITIONS[second]
        return math.hypot(x2 - x1, y2 - y1)

    def astar(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        scale = self._heuristic_scale()

        def heuristic(node: str) -> float:
            return self._distance(node, goal) * scale

        frontier: list[tuple[float, int, int, str, list[str]]] = [
            (heuristic(start), 0, 0, start, [start])
        ]
        best_cost: dict[tuple[str, int], int] = {(start, 0): 0}
        visited_order: list[str] = []
        trace: list[dict[str, Any]] = []
        counter = 1

        while frontier:
            f_score, cost, _, node, path = heapq.heappop(frontier)
            depth = len(path) - 1
            state = (node, depth)
            if cost != best_cost.get(state):
                continue
            visited_order.append(node)
            trace.append(
                {
                    "action": "expand",
                    "node": node,
                    "g": round(cost, 3),
                    "h": round(heuristic(node), 3),
                    "f": round(f_score, 3),
                }
            )

            if node == goal:
                return SearchOutcome(
                    True,
                    path,
                    cost,
                    visited_order,
                    trace,
                    metadata={"heuristic_scale": round(scale, 4)},
                )

            if depth >= max_stops:
                continue

            for neighbor, edge_cost in self.graph[node].items():
                if neighbor in path:
                    continue
                new_cost = cost + edge_cost
                if new_cost > time_limit:
                    continue
                next_state = (neighbor, depth + 1)
                if new_cost < best_cost.get(next_state, math.inf):
                    best_cost[next_state] = new_cost
                    h = heuristic(neighbor)
                    heapq.heappush(
                        frontier,
                        (new_cost + h, new_cost, counter, neighbor, path + [neighbor]),
                    )
                    counter += 1
                    trace.append(
                        {
                            "action": "score",
                            "from": node,
                            "node": neighbor,
                            "g": new_cost,
                            "h": round(h, 3),
                            "f": round(new_cost + h, 3),
                        }
                    )

        return SearchOutcome(False, visited_order=visited_order, trace=trace, warning="A* found no feasible route.")

    def hill_climbing(
        self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any
    ) -> SearchOutcome:
        self.validate_nodes(start, goal)
        path = [start]
        total_time = 0
        visited_order = [start]
        trace: list[dict[str, Any]] = []

        while path[-1] != goal and len(path) - 1 < max_stops:
            current = path[-1]
            current_h = self._distance(current, goal)
            candidates = []
            for neighbor, edge_cost in self.graph[current].items():
                if neighbor in path or total_time + edge_cost > time_limit:
                    continue
                candidates.append((self._distance(neighbor, goal), edge_cost, neighbor))

            if not candidates:
                return SearchOutcome(
                    False,
                    path,
                    total_time,
                    visited_order,
                    trace,
                    warning="Hill climbing reached a dead end under the current constraints.",
                )

            next_h, edge_cost, next_node = min(candidates)
            trace.append(
                {
                    "action": "choose_best_neighbor",
                    "node": current,
                    "current_h": round(current_h, 2),
                    "next": next_node,
                    "next_h": round(next_h, 2),
                }
            )

            if next_h >= current_h and next_node != goal:
                return SearchOutcome(
                    False,
                    path,
                    total_time,
                    visited_order,
                    trace,
                    warning="Hill climbing stopped at a local optimum.",
                )

            path.append(next_node)
            visited_order.append(next_node)
            total_time += edge_cost

        if path[-1] == goal:
            return SearchOutcome(True, path, total_time, visited_order, trace)
        return SearchOutcome(
            False,
            path,
            total_time,
            visited_order,
            trace,
            warning="The stop limit was reached before the destination.",
        )

    def csp(self, start: str, goal: str, time_limit: int, max_stops: int, **_: Any) -> SearchOutcome:
        self.validate_nodes(start, goal)
        visited_order: list[str] = []
        trace: list[dict[str, Any]] = []
        best_path: list[str] | None = None
        best_time = math.inf
        assignments_checked = 0

        def backtrack(path: list[str], total: int) -> None:
            nonlocal best_path, best_time, assignments_checked
            node = path[-1]
            visited_order.append(node)
            assignments_checked += 1
            trace.append(
                {
                    "action": "assign",
                    "node": node,
                    "time": total,
                    "remaining_stops": max_stops - (len(path) - 1),
                }
            )

            if total >= best_time:
                return
            if node == goal:
                best_path = path.copy()
                best_time = total
                trace.append({"action": "solution", "path": path, "time": total})
                return
            if len(path) - 1 >= max_stops:
                return

            for neighbor, edge_cost in sorted(self.graph[node].items(), key=lambda item: item[1]):
                if neighbor in path:
                    continue
                new_total = total + edge_cost
                if new_total <= time_limit:
                    backtrack(path + [neighbor], new_total)
                else:
                    trace.append(
                        {
                            "action": "prune",
                            "from": node,
                            "node": neighbor,
                            "reason": "time_limit",
                        }
                    )

        backtrack([start], 0)
        if best_path:
            return SearchOutcome(
                True,
                best_path,
                int(best_time),
                visited_order,
                trace,
                metadata={"assignments_checked": assignments_checked},
            )
        return SearchOutcome(
            False,
            visited_order=visited_order,
            trace=trace,
            warning="No simple route satisfies all selected constraints.",
            metadata={"assignments_checked": assignments_checked},
        )

    def genetic(
        self,
        start: str,
        goal: str,
        time_limit: int,
        max_stops: int,
        seed: int = 42,
        **_: Any,
    ) -> SearchOutcome:
        self.validate_nodes(start, goal)
        rng = random.Random(seed)
        population_size = 18
        generations = 35
        mutation_rate = 0.55
        trace: list[dict[str, Any]] = []
        visited_order: list[str] = []

        def random_path(prefix: list[str] | None = None) -> list[str] | None:
            base = (prefix or [start]).copy()

            def walk(path: list[str], total: int) -> list[str] | None:
                node = path[-1]
                visited_order.append(node)
                if node == goal:
                    return path
                if len(path) - 1 >= max_stops:
                    return None
                neighbors = list(self.graph[node].items())
                rng.shuffle(neighbors)
                neighbors.sort(key=lambda item: self._distance(item[0], goal) + rng.random() * 18)
                for neighbor, edge_cost in neighbors:
                    if neighbor in path or total + edge_cost > time_limit:
                        continue
                    result = walk(path + [neighbor], total + edge_cost)
                    if result:
                        return result
                return None

            return walk(base, self.path_time(base))

        def fitness(path: list[str]) -> float:
            if not path or path[-1] != goal or not self._within_constraints(path, time_limit, max_stops):
                return 0.0
            return 1.0 / max(1, self.path_time(path))

        def crossover(first: list[str], second: list[str]) -> list[str]:
            common = [node for node in first[1:-1] if node in second[1:-1]]
            if not common:
                return first.copy() if fitness(first) >= fitness(second) else second.copy()
            pivot = rng.choice(common)
            child = first[: first.index(pivot)] + second[second.index(pivot) :]
            if len(child) != len(set(child)):
                return first.copy()
            return child

        def mutate(path: list[str]) -> list[str]:
            if len(path) <= 2:
                return path
            cut = rng.randint(1, len(path) - 1)
            prefix = path[:cut]
            candidate = random_path(prefix)
            return candidate or path

        population: list[list[str]] = []
        attempts = 0
        while len(population) < population_size and attempts < 240:
            attempts += 1
            candidate = random_path()
            if candidate and fitness(candidate) > 0:
                population.append(candidate)

        if not population:
            return SearchOutcome(
                False,
                visited_order=visited_order,
                trace=trace,
                warning="The genetic population could not produce a feasible initial route.",
            )

        best = max(population, key=fitness)
        for generation in range(generations):
            population.sort(key=fitness, reverse=True)
            if fitness(population[0]) > fitness(best):
                best = population[0].copy()

            trace.append(
                {
                    "action": "generation",
                    "generation": generation + 1,
                    "best_time": self.path_time(population[0]),
                    "fitness": round(fitness(population[0]), 6),
                    "diversity": len({tuple(path) for path in population}),
                }
            )

            elite_count = max(2, population_size // 4)
            elites = population[:elite_count]
            next_generation = [path.copy() for path in elites]

            while len(next_generation) < population_size:
                parent_a = rng.choice(elites)
                parent_b = rng.choice(population[: max(elite_count + 2, len(population) // 2)])
                child = crossover(parent_a, parent_b)
                if rng.random() < mutation_rate:
                    child = mutate(child)
                if fitness(child) > 0:
                    next_generation.append(child)
                else:
                    replacement = random_path()
                    if replacement:
                        next_generation.append(replacement)

            population = next_generation[:population_size]

        population.sort(key=fitness, reverse=True)
        if fitness(population[0]) > fitness(best):
            best = population[0]

        return SearchOutcome(
            True,
            best,
            self.path_time(best),
            visited_order,
            trace,
            metadata={
                "seed": seed,
                "generations": generations,
                "population_size": population_size,
                "fitness": round(fitness(best), 6),
            },
        )

    def run(
        self,
        algorithm: str,
        start: str,
        goal: str,
        time_limit: int,
        max_stops: int,
        seed: int = 42,
    ) -> tuple[SearchOutcome, float]:
        methods: dict[str, Callable[..., SearchOutcome]] = {
            "bfs": self.bfs,
            "dfs": self.dfs,
            "ucs": self.ucs,
            "iddfs": self.iddfs,
            "astar": self.astar,
            "hill_climbing": self.hill_climbing,
            "csp": self.csp,
            "genetic": self.genetic,
        }
        if algorithm not in methods:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        started = time.perf_counter()
        outcome = methods[algorithm](
            start=start,
            goal=goal,
            time_limit=time_limit,
            max_stops=max_stops,
            seed=seed,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return outcome, elapsed_ms
