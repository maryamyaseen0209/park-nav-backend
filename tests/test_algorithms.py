from app.algorithms import ParkSearchEngine


ENGINE = ParkSearchEngine()
START = "Saddar"
GOAL = "Nisar Shaheed Park"


def assert_valid(outcome):
    assert outcome.found is True
    assert outcome.path[0] == START
    assert outcome.path[-1] == GOAL
    assert outcome.total_time == ENGINE.path_time(outcome.path)
    assert len(outcome.path) == len(set(outcome.path))


def test_ucs_returns_optimal_weighted_route():
    outcome = ENGINE.ucs(START, GOAL, time_limit=100, max_stops=10)
    assert_valid(outcome)
    assert outcome.total_time == 45
    assert outcome.path == [
        "Saddar",
        "Jahangir Park",
        "Shaheed Benazir Bhutto Park",
        "Nisar Shaheed Park",
    ]


def test_astar_matches_ucs_cost():
    ucs = ENGINE.ucs(START, GOAL, time_limit=100, max_stops=10)
    astar = ENGINE.astar(START, GOAL, time_limit=100, max_stops=10)
    assert_valid(astar)
    assert astar.total_time == ucs.total_time


def test_bfs_finds_fewest_transition_route():
    outcome = ENGINE.bfs(START, GOAL, time_limit=100, max_stops=10)
    assert_valid(outcome)
    assert len(outcome.path) - 1 == 3


def test_iddfs_finds_shallow_route():
    outcome = ENGINE.iddfs(START, GOAL, time_limit=100, max_stops=10)
    assert_valid(outcome)
    assert outcome.metadata["depth_found"] == 3


def test_csp_respects_constraints():
    outcome = ENGINE.csp(START, GOAL, time_limit=50, max_stops=4)
    assert_valid(outcome)
    assert outcome.total_time <= 50
    assert len(outcome.path) - 1 <= 4


def test_genetic_is_reproducible_and_valid():
    first = ENGINE.genetic(START, GOAL, time_limit=100, max_stops=10, seed=7)
    second = ENGINE.genetic(START, GOAL, time_limit=100, max_stops=10, seed=7)
    assert_valid(first)
    assert first.path == second.path
    assert first.total_time == second.total_time


def test_time_limit_can_make_route_impossible():
    outcome = ENGINE.ucs(START, GOAL, time_limit=20, max_stops=10)
    assert outcome.found is False
