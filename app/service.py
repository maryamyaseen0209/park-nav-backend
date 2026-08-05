from __future__ import annotations

from .algorithms import ParkSearchEngine, SearchOutcome
from .data import ALGORITHM_CATALOG
from .models import RouteResult


CATALOG_BY_ID = {item["id"]: item for item in ALGORITHM_CATALOG}


class RouteService:
    def __init__(self) -> None:
        self.engine = ParkSearchEngine()

    def _explain(self, algorithm: str, outcome: SearchOutcome) -> str:
        if not outcome.found:
            return outcome.warning or "No valid route could be produced for the selected constraints."

        path_text = " → ".join(outcome.path)
        explanations = {
            "bfs": f"BFS selected {path_text} because it reaches the destination using the fewest graph transitions among valid shallow routes.",
            "dfs": f"DFS returned {path_text} after following one branch deeply before backtracking. It is a valid exploratory route, not necessarily the fastest.",
            "ucs": f"UCS selected {path_text} by always expanding the currently cheapest accumulated travel-time option.",
            "iddfs": f"IDDFS found {path_text} at the first depth where the destination became reachable, while respecting the route constraints.",
            "astar": f"A* selected {path_text} by combining accumulated travel time with an admissible geometric estimate toward the destination.",
            "hill_climbing": f"Hill climbing followed {path_text} by repeatedly choosing the neighbouring park that appeared closest to the goal.",
            "csp": f"The constraint solver selected {path_text} after backtracking through simple routes and pruning assignments that violated time or stop limits.",
            "genetic": f"The genetic algorithm evolved {path_text} through selection, crossover, and mutation, favouring lower-time valid routes.",
        }
        return explanations[algorithm]

    def build_result(self, algorithm: str, outcome: SearchOutcome, runtime_ms: float) -> RouteResult:
        info = CATALOG_BY_ID[algorithm]
        optimality = {
            "bfs": "Optimal for number of transitions, not weighted travel time",
            "dfs": "No optimality guarantee",
            "ucs": "Optimal for non-negative travel-time weights",
            "iddfs": "Optimal for shallowest depth under uniform step cost",
            "astar": "Optimal with this admissible heuristic and non-negative weights",
            "hill_climbing": "No global optimality guarantee",
            "csp": "Best feasible route found by exhaustive constrained backtracking",
            "genetic": "Approximate stochastic solution; reproducible with the same seed",
        }[algorithm]

        return RouteResult(
            algorithm=algorithm,
            algorithm_name=info["name"],
            found=outcome.found,
            path=outcome.path,
            total_time=outcome.total_time,
            stops=max(0, len(outcome.path) - 1),
            visited_order=outcome.visited_order,
            explored_count=len(outcome.visited_order),
            runtime_ms=round(runtime_ms, 4),
            objective=info["objective"],
            optimality=optimality,
            explanation=self._explain(algorithm, outcome),
            warning=outcome.warning,
            trace=outcome.trace[:160],
            metadata=outcome.metadata,
        )

    def search(
        self,
        algorithm: str,
        start: str,
        goal: str,
        time_limit: int,
        max_stops: int,
        seed: int,
    ) -> RouteResult:
        outcome, runtime_ms = self.engine.run(
            algorithm=algorithm,
            start=start,
            goal=goal,
            time_limit=time_limit,
            max_stops=max_stops,
            seed=seed,
        )
        return self.build_result(algorithm, outcome, runtime_ms)
