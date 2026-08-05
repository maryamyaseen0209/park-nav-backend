from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data import ALGORITHM_CATALOG, GRAPH, PARKS
from .models import CompareRequest, NetworkResponse, RouteRequest, RouteResult
from .service import RouteService


service = RouteService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="ParkNova AI API",
    version="1.0.0",
    description="A production-style API for comparing classical AI search and optimization algorithms on a weighted park network.",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    return {
        "product": "ParkNova AI",
        "status": "online",
        "docs": "/docs",
        "dataset_note": "Academic weighted graph; not live GPS navigation.",
    }


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "healthy", "parks": len(PARKS), "algorithms": len(ALGORITHM_CATALOG)}


@app.get("/api/parks", tags=["Network"])
def parks():
    return PARKS


@app.get("/api/network", response_model=NetworkResponse, tags=["Network"])
def network():
    edges = []
    seen = set()
    for source, neighbors in GRAPH.items():
        for target, minutes in neighbors.items():
            key = tuple(sorted((source, target)))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": source, "target": target, "minutes": minutes})
    return {"parks": PARKS, "edges": edges}


@app.get("/api/algorithms", tags=["Algorithms"])
def algorithms():
    return ALGORITHM_CATALOG


@app.post("/api/routes/search", response_model=RouteResult, tags=["Routes"])
def search_route(payload: RouteRequest):
    try:
        return service.search(
            algorithm=payload.algorithm.value,
            start=payload.start,
            goal=payload.goal,
            time_limit=payload.time_limit,
            max_stops=payload.max_stops,
            seed=payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/routes/compare", response_model=list[RouteResult], tags=["Routes"])
def compare_routes(payload: CompareRequest):
    algorithms_to_run = payload.algorithms or [item["id"] for item in ALGORITHM_CATALOG]
    results = []
    for algorithm in algorithms_to_run:
        algorithm_id = algorithm.value if hasattr(algorithm, "value") else str(algorithm)
        try:
            results.append(
                service.search(
                    algorithm=algorithm_id,
                    start=payload.start,
                    goal=payload.goal,
                    time_limit=payload.time_limit,
                    max_stops=payload.max_stops,
                    seed=payload.seed,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return results
