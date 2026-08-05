from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Algorithm(str, Enum):
    BFS = "bfs"
    DFS = "dfs"
    UCS = "ucs"
    IDDFS = "iddfs"
    ASTAR = "astar"
    HILL_CLIMBING = "hill_climbing"
    CSP = "csp"
    GENETIC = "genetic"


class RouteRequest(BaseModel):
    start: str
    goal: str
    algorithm: Algorithm = Algorithm.ASTAR
    time_limit: int = Field(default=120, ge=1, le=1000)
    max_stops: int = Field(default=10, ge=1, le=30)
    seed: int = Field(default=42, ge=0, le=1_000_000)

    @field_validator("goal")
    @classmethod
    def different_goal(cls, value: str, info):
        start = info.data.get("start")
        if start and value == start:
            raise ValueError("Start and destination must be different parks")
        return value


class CompareRequest(BaseModel):
    start: str
    goal: str
    time_limit: int = Field(default=120, ge=1, le=1000)
    max_stops: int = Field(default=10, ge=1, le=30)
    seed: int = Field(default=42, ge=0, le=1_000_000)
    algorithms: list[Algorithm] | None = None


class RouteResult(BaseModel):
    algorithm: str
    algorithm_name: str
    found: bool
    path: list[str]
    total_time: int | None
    stops: int
    visited_order: list[str]
    explored_count: int
    runtime_ms: float
    objective: str
    optimality: str
    explanation: str
    warning: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NetworkResponse(BaseModel):
    parks: list[dict[str, Any]]
    edges: list[dict[str, Any]]
