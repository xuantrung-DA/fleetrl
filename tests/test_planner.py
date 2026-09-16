"""Exact route and physically meaningful reservation regression tests."""

import random

import pytest

from fleetrl.planner import GridPlanner, ReservationTable


def test_shortest_path_obstacles_and_unreachable():
    planner = GridPlanner(5, 4, blocked={(2, 0), (2, 1), (2, 2)})
    path = planner.shortest_path((0, 0), (4, 0))
    assert len(path) - 1 == 10
    assert planner.distance((0, 0), (4, 0)) == 10
    assert (2, 3) in path
    planner.set_blocked((2, 3))
    assert planner.shortest_path((0, 0), (4, 0)) is None
    assert planner.distance((0, 0), (4, 0)) is None
    planner.set_blocked((2, 3), False)
    assert planner.distance((0, 0), (4, 0)) == 10


def test_dynamic_detour_and_blocked_edge_cache_invalidation():
    planner = GridPlanner(4, 3)
    assert planner.distance((0, 0), (3, 0)) == 3
    path = planner.shortest_path((0, 0), (3, 0), blocked_extra={(1, 0)})
    assert len(path) - 1 == 5
    planner.set_edge_blocked((1, 0), (2, 0))
    assert planner.distance((0, 0), (3, 0)) == 5
    planner.set_edge_blocked((2, 0), (1, 0), False)
    assert planner.distance((0, 0), (3, 0)) == 3
    path = planner.shortest_path((0, 0), (3, 0), blocked_edges={((2, 0), (1, 0))})
    assert len(path) - 1 == 5


def test_vertex_collision_and_head_on_swap_rejected():
    table = ReservationTable({"a": (0, 0), "b": (2, 0)})
    assert table.try_reserve_move("a", (0, 0), (1, 0), 0, 2)
    assert not table.try_reserve_move("b", (2, 0), (1, 0), 0, 1)
    table.advance(2)
    assert not table.try_reserve_move("a", (1, 0), (2, 0), 2, 1)
    assert not table.try_reserve_move("b", (2, 0), (1, 0), 2, 1)
    table.assert_safe()


def test_origin_held_until_arrival_no_following_through_robot():
    table = ReservationTable({1: (1, 0), 2: (0, 0)})
    assert table.try_reserve_move(1, (1, 0), (2, 0), 0, 3)
    assert not table.try_reserve_move(2, (0, 0), (1, 0), 0, 1)
    assert table.advance(2) == {}
    assert not table.try_reserve_move(2, (0, 0), (1, 0), 2, 1)
    assert table.advance(3) == {1: (2, 0)}
    assert table.try_reserve_move(2, (0, 0), (1, 0), 3, 1)
    assert table.advance(4) == {2: (1, 0)}
    table.assert_safe()


def test_pending_target_and_breakdown_occupancy_are_preserved():
    table = ReservationTable({"moving": (0, 0), "broken": (3, 0)})
    table.try_reserve_move("moving", (0, 0), (1, 0), 0, 3)
    assert table.occupied_cells() == {(0, 0), (1, 0), (3, 0)}
    assert table.occupied_cells("moving") == {(3, 0)}
    assert table.pending_move("moving").end_tick == 3
    assert not table.is_available((3, 0), tick=1000)
    assert not table.is_available((1, 0), tick=1000)
    assert table.is_available((0, 0), tick=3)
    with pytest.raises(ValueError, match="in-flight"):
        table.release("moving")


def test_timed_path_waits_for_committed_origin_release():
    planner = GridPlanner(3, 2, blocked={(0, 1), (2, 1)})
    table = ReservationTable({"a": (0, 0), "b": (1, 0)})
    table.try_reserve_move("b", (1, 0), (1, 1), 0, 3)
    path = planner.timed_path((0, 0), (2, 0), 0, 1, table, "a")
    assert path == [(0, (0, 0)), (1, (0, 0)), (2, (0, 0)), (3, (0, 0)), (4, (1, 0)), (5, (2, 0))]
    for (tick, a), (next_tick, b) in zip(path, path[1:]):
        assert table.can_traverse("a", a, b, tick, next_tick - tick)


def test_timed_path_bounded_prefix_and_stationary_corridor_block():
    planner = GridPlanner(20, 1)
    table = ReservationTable({0: (0, 0)})
    path = planner.timed_path((0, 0), (19, 0), 0, 3, table, 0, horizon_ticks=12)
    assert path[-1] == (12, (4, 0))
    table.register(1, (1, 0))
    assert planner.timed_path((0, 0), (19, 0), 0, 1, table, 0) is None


def test_opposing_corridor_routes_wait_without_teleporting():
    planner = GridPlanner(5, 1)
    table = ReservationTable({"a": (0, 0), "b": (4, 0)})
    goals = {"a": (4, 0), "b": (0, 0)}
    for tick in range(15):
        table.advance(tick)
        for robot in ("a", "b"):
            if table.pending_move(robot):
                continue
            path = planner.timed_path(
                table.position(robot), goals[robot], tick, 1, table, robot, horizon_ticks=8
            )
            if path and len(path) > 1 and path[1][1] != path[0][1]:
                table.try_reserve_move(robot, path[0][1], path[1][1], tick, 1)
        table.assert_safe()
    assert table.position("a")[0] < table.position("b")[0]


def test_heterogeneous_random_traffic_invariants():
    rng = random.Random(19)
    planner = GridPlanner(10, 8)
    cells = rng.sample([(x, y) for x in range(10) for y in range(8)], 20)
    table = ReservationTable(dict(enumerate(cells)))
    accepted = 0
    for tick in range(300):
        previous = table.positions
        arrivals = table.advance(tick)
        for robot, cell in arrivals.items():
            assert abs(previous[robot][0] - cell[0]) + abs(previous[robot][1] - cell[1]) == 1
        order = list(range(20))
        rng.shuffle(order)
        for robot in order:
            if table.pending_move(robot):
                continue
            current = table.position(robot)
            target = rng.choice(planner.neighbors(current))
            accepted += table.try_reserve_move(robot, current, target, tick, 1 + robot % 3)
        table.assert_safe()
    assert accepted > 1000


def test_invalid_registration_geometry_and_time():
    table = ReservationTable({0: (0, 0)})
    with pytest.raises(ValueError):
        table.register(1, (0, 0))
    with pytest.raises(ValueError):
        table.try_reserve_move(0, (0, 0), (2, 0), 0, 1)
    with pytest.raises(ValueError):
        table.try_reserve_move(0, (0, 0), (1, 0), 1, 1)
    table.advance(2)
    with pytest.raises(ValueError):
        table.advance(1)


def test_immediate_cancel_keeps_origin_and_releases_only_destination():
    table = ReservationTable({"a": (0, 0), "b": (2, 0)})
    assert table.try_reserve_move("a", (0, 0), (1, 0), 0, 3)
    cancelled = table.cancel_move("a")
    assert cancelled.target == (1, 0)
    assert table.position("a") == (0, 0)
    assert table.pending_move("a") is None
    assert not table.is_available((0, 0), robot="b")
    assert table.try_reserve_move("b", (2, 0), (1, 0), 0, 1)
    assert table.advance(1) == {"b": (1, 0)}
    assert table.cancel_move("a") is None
    table.assert_safe()


def test_cancel_after_edge_entry_is_forbidden():
    table = ReservationTable({"a": (0, 0)})
    table.try_reserve_move("a", (0, 0), (1, 0), 0, 3)
    table.advance(1)
    with pytest.raises(ValueError, match="physical time"):
        table.cancel_move("a")
    assert table.occupied_cells() == {(0, 0), (1, 0)}
    assert table.advance(3) == {"a": (1, 0)}
    table.assert_safe()


def _corridor_planner():
    # Two three-cell bays joined by a three-cell, one-cell-wide corridor.
    free = {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1), (4, 0), (4, 1), (4, 2)}
    return GridPlanner(5, 3, blocked={(x, y) for x in range(5) for y in range(3)} - free)


def test_corridor_opposing_entry_is_rejected_before_robots_meet():
    planner = _corridor_planner()
    assert len(planner.corridors) == 1
    assert planner.corridors[0].interior == ((1, 1), (2, 1), (3, 1))
    table = ReservationTable({"east": (0, 1), "west": (4, 1)})
    assert planner.corridor_move_allowed("east", (0, 1), (1, 1), table)
    assert table.try_reserve_move("east", (0, 1), (1, 1), 0, 2)
    # This opposite destination is physically free, so the directional resource
    # adds a real constraint beyond local cell/edge reservations.
    assert table.can_traverse("west", (4, 1), (3, 1), 0, 2)
    assert not planner.corridor_move_allowed("west", (4, 1), (3, 1), table)


def test_corridor_same_direction_following_with_free_gap():
    planner = _corridor_planner()
    table = ReservationTable({"leader": (2, 1), "follower": (0, 1)})
    assert planner.corridor_move_allowed("leader", (2, 1), (3, 1), table)
    assert table.try_reserve_move("leader", (2, 1), (3, 1), 0, 2)
    assert planner.corridor_move_allowed("follower", (0, 1), (1, 1), table)
    assert table.try_reserve_move("follower", (0, 1), (1, 1), 0, 1)
    table.assert_safe()


def test_corridor_unknown_stationary_direction_blocks_entry_but_not_exit():
    planner = _corridor_planner()
    table = ReservationTable({"resident": (3, 1), "entrant": (0, 1)})
    assert not planner.corridor_move_allowed("entrant", (0, 1), (1, 1), table)
    assert planner.corridor_move_allowed("resident", (3, 1), (4, 1), table)


def test_corridor_unlocks_after_physical_exit_and_has_no_stale_state():
    planner = _corridor_planner()
    table = ReservationTable({"resident": (3, 1)})
    assert table.try_reserve_move("resident", (3, 1), (4, 1), 0, 2)
    assert not planner.corridor_move_allowed("reverse", (4, 1), (3, 1), table)
    table.advance(1)
    assert not planner.corridor_move_allowed("reverse", (4, 1), (3, 1), table)
    table.advance(2)
    # Only resource availability is queried; actual endpoint occupancy remains
    # the separate ReservationTable commitment check.
    assert planner.corridor_move_allowed("reverse", (4, 1), (3, 1), table)


def test_corridor_topology_rebuild_and_corner_chain():
    planner = _corridor_planner()
    assert len(planner.corridors) == 1
    planner.set_blocked((2, 1))
    assert planner.corridors == ()
    planner.set_blocked((2, 1), False)
    assert len(planner.corridors) == 1
    corner = GridPlanner(3, 3, blocked={(0, 1), (0, 2), (1, 1), (1, 2)})
    assert corner.corridors[0].cells == ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))


def test_timed_path_extra_blocks_avoid_port_transit_and_allow_start_egress():
    planner = GridPlanner(4, 3)
    table = ReservationTable({"egress": (0, 0)})
    path = planner.timed_path(
        (0, 0), (3, 0), 0, 1, table, "egress", blocked_extra={(0, 0), (1, 0), (2, 0)}
    )
    assert path[0] == (0, (0, 0))
    assert path[-1] == (5, (3, 0))
    assert all(cell not in {(1, 0), (2, 0)} for _, cell in path)
    assert planner.timed_path((0, 0), (3, 0), 0, 1, table, "egress", blocked_extra={(3, 0)}) is None
