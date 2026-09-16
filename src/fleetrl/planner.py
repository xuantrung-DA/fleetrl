"""Deterministic grid routing and conservative physical movement reservations.

The simulator advances the reservation clock once per physical tick. A committed
move holds *both* endpoint cells until arrival; this deliberately disallows
following another robot into its cell during the same edge traversal. Stationary
robots occupy their cells indefinitely. ``timed_path`` only plans against known
departures, so it is safe but incomplete: a tight corridor can require the
simulator's explicit yielding/recovery policy. No method teleports a robot.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import count
from typing import Hashable, Iterable, Mapping

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]
RobotID = Hashable


def _edge(a: Cell, b: Cell) -> Edge:
    """Canonical undirected edge; blocking forbids both travel directions."""
    return (a, b) if a <= b else (b, a)


def _adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


@dataclass(frozen=True)
class MoveReservation:
    """A physical edge traversal with arrival at ``end_tick``."""

    robot: RobotID
    origin: Cell
    target: Cell
    start_tick: int
    end_tick: int

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass(frozen=True)
class NarrowCorridor:
    """Ordered degree-2 interior between two junction or dead-end endpoints."""

    id: int
    endpoints: tuple[Cell, Cell]
    interior: tuple[Cell, ...]

    @property
    def cells(self) -> tuple[Cell, ...]:
        return (self.endpoints[0], *self.interior, self.endpoints[1])


class ReservationTable:
    """Single source of truth for physical cells and committed edge intervals.

    Register every robot before simulation. Use ``advance(tick)`` before moving
    robots at that tick, then ``try_reserve_move`` sequentially in a rotating
    robot order. Reservation acceptance is atomic. Returned mappings are copies
    so external code cannot silently break the ownership invariants.
    """

    def __init__(self, positions: Mapping[RobotID, Cell] | None = None) -> None:
        self.tick = 0
        self._positions: dict[RobotID, Cell] = {}
        self._owners: dict[Cell, RobotID] = {}
        self._moves: dict[RobotID, MoveReservation] = {}
        self._targets: dict[Cell, RobotID] = {}
        for robot, cell in (positions or {}).items():
            self.register(robot, cell)

    @property
    def positions(self) -> dict[RobotID, Cell]:
        """Current settled/origin cells; an in-flight robot stays at its origin."""
        return dict(self._positions)

    @property
    def moves(self) -> dict[RobotID, MoveReservation]:
        return dict(self._moves)

    @property
    def reservations(self) -> dict[RobotID, MoveReservation]:
        return self.moves

    def position(self, robot: RobotID) -> Cell:
        return self._positions[robot]

    def pending_move(self, robot: RobotID) -> MoveReservation | None:
        return self._moves.get(robot)

    def register(self, robot: RobotID, cell: Cell) -> None:
        """Register a unique robot on an unoccupied cell (same-cell is idempotent)."""
        cell = tuple(cell)
        if robot in self._positions:
            if self._positions[robot] == cell and robot not in self._moves:
                return
            raise ValueError(f"Robot {robot!r} is already registered")
        if not self.is_available(cell):
            raise ValueError(f"Cell {cell} is occupied or reserved")
        self._positions[robot] = cell
        self._owners[cell] = robot

    def occupied_cells(self, robot_exclude: RobotID | None = None) -> set[Cell]:
        """All held origins and destinations, optionally ignoring one robot."""
        return {
            cell
            for cell, robot in (*self._owners.items(), *self._targets.items())
            if robot != robot_exclude
        }

    def is_available(
        self,
        cell: Cell,
        robot: RobotID | None = None,
        tick: int | None = None,
    ) -> bool:
        """Whether a cell is free at a time using only committed departures.

        Targets are held from reservation until an eventual later departure.
        An origin becomes free at its committed arrival time. No future intent
        is assumed for a stationary robot, including stopped/stale robots.
        """
        when = self.tick if tick is None else tick
        target_owner = self._targets.get(cell)
        if target_owner is not None and target_owner != robot:
            return False
        owner = self._owners.get(cell)
        if owner is None or owner == robot:
            return True
        move = self._moves.get(owner)
        return move is not None and when >= move.end_tick

    def can_traverse(
        self,
        robot: RobotID,
        origin: Cell,
        target: Cell,
        start_tick: int,
        duration_ticks: int,
    ) -> bool:
        """Check both cell holds and opposite/same-edge interval conflicts.

        This is a planning query: ``origin`` need not be the robot's current
        cell. Actual commitment also checks registration, origin, and clock.
        Waiting (origin == target) is allowed here, but not a physical move.
        """
        if duration_ticks <= 0 or start_tick < self.tick:
            return False
        if origin != target and not _adjacent(origin, target):
            return False
        end_tick = start_tick + duration_ticks
        # No already committed future move can occupy a new cell later without
        # its target being held now; endpoints therefore suffice for intervals.
        if not all(
            self.is_available(cell, robot, when)
            for cell in (origin, target)
            for when in (start_tick, end_tick)
        ):
            return False
        if origin != target:
            edge = _edge(origin, target)
            for other, move in self._moves.items():
                if other != robot and _edge(move.origin, move.target) == edge:
                    if start_tick < move.end_tick and move.start_tick < end_tick:
                        return False
        return True

    def try_reserve_move(
        self,
        robot: RobotID,
        origin: Cell,
        target: Cell,
        start_tick: int,
        duration_ticks: int,
    ) -> bool:
        """Commit an adjacent move starting now, returning False on contention.

        Invalid times/geometry and unregistered robots are programming errors.
        Advance the table first: future intentions must not reserve physical
        space before their execution tick.
        """
        if robot not in self._positions:
            raise KeyError(f"Unregistered robot {robot!r}")
        if not isinstance(duration_ticks, int) or duration_ticks <= 0:
            raise ValueError("duration_ticks must be a positive integer")
        if start_tick != self.tick:
            raise ValueError("Move start must equal the reservation clock; call advance first")
        if not _adjacent(origin, target):
            raise ValueError("A move must traverse exactly one grid edge")
        if robot in self._moves or self._positions[robot] != origin:
            return False
        if not self.can_traverse(robot, origin, target, start_tick, duration_ticks):
            return False
        self._moves[robot] = MoveReservation(
            robot,
            origin,
            target,
            start_tick,
            start_tick + duration_ticks,
        )
        self._targets[target] = robot
        return True

    def advance(self, tick: int) -> dict[RobotID, Cell]:
        """Complete all arrivals up to tick and return robot -> arrival cell."""
        if not isinstance(tick, int) or tick < self.tick:
            raise ValueError("Reservation time must be monotonic integer ticks")
        self.tick = tick
        arrivals: dict[RobotID, Cell] = {}
        for robot, move in list(self._moves.items()):
            if move.end_tick <= tick:
                if self._owners.get(move.origin) != robot:
                    raise RuntimeError("Reservation origin ownership was corrupted")
                if move.target in self._owners:
                    raise RuntimeError("Arrival would cause a vertex collision")
                del self._owners[move.origin]
                del self._targets[move.target]
                self._owners[move.target] = robot
                self._positions[robot] = move.target
                del self._moves[robot]
                arrivals[robot] = move.target
        return arrivals

    def cancel_move(self, robot: RobotID) -> MoveReservation | None:
        """Rollback an unexecuted move at its commitment tick, keeping origin.

        Used when the simulator's final energy validation rejects entry after
        an otherwise successful reservation. Once the clock advances, physical
        movement has begun and cancellation is forbidden. A stopped moving
        robot must complete its owned edge before holding its destination.
        """
        move = self._moves.get(robot)
        if move is None:
            return None
        if self.tick != move.start_tick:
            raise ValueError("Cannot cancel a move after physical time has advanced")
        del self._targets[move.target]
        del self._moves[robot]
        return move

    def release(self, robot: RobotID) -> None:
        """Remove a stationary robot explicitly; never use for a stopped robot.

        Breakdown/stale robots must retain occupancy. In-flight removal is
        forbidden, since releasing its cells would invalidate physical safety.
        """
        if robot in self._moves:
            raise ValueError("Cannot release an in-flight robot")
        cell = self._positions.pop(robot)
        del self._owners[cell]

    def assert_safe(self) -> None:
        """Raise if bookkeeping could permit a vertex or edge-swap collision."""
        if len(set(self._positions.values())) != len(self._positions):
            raise AssertionError("Two robots occupy one cell")
        if self._owners != {cell: robot for robot, cell in self._positions.items()}:
            raise AssertionError("Owner and position mappings differ")
        expected_targets = {m.target: robot for robot, m in self._moves.items()}
        if len(expected_targets) != len(self._moves) or expected_targets != self._targets:
            raise AssertionError("Destination reservation conflict")
        for robot, move in self._moves.items():
            if self._positions[robot] != move.origin or not _adjacent(move.origin, move.target):
                raise AssertionError("Invalid in-flight move")
            if move.target in self._owners and self._owners[move.target] != robot:
                raise AssertionError("In-flight target intersects occupied cell")


class GridPlanner:
    """Four-neighbor routes with bounded caches and a rolling time-space A*."""

    def __init__(
        self,
        width: int,
        height: int,
        blocked: Iterable[Cell] = (),
        blocked_edges: Iterable[Edge] = (),
        cache_size: int = 128,
    ) -> None:
        if width <= 0 or height <= 0 or cache_size <= 0:
            raise ValueError("Grid dimensions and cache_size must be positive")
        self.width, self.height = width, height
        self.blocked = {tuple(cell) for cell in blocked}
        self.blocked_edges = {_edge(tuple(a), tuple(b)) for a, b in blocked_edges}
        if any(not self.in_bounds(cell) for cell in self.blocked):
            raise ValueError("Blocked cell outside grid")
        if any(
            not _adjacent(a, b) or not self.in_bounds(a) or not self.in_bounds(b)
            for a, b in self.blocked_edges
        ):
            raise ValueError("Blocked edge must join adjacent in-bounds cells")
        self.cache_size = cache_size
        self._distance_cache: OrderedDict[Cell, dict[Cell, int]] = OrderedDict()
        self._corridors: tuple[NarrowCorridor, ...] | None = None
        self._corridor_by_cell: dict[Cell, NarrowCorridor] = {}
        self._corridor_indices: dict[int, dict[Cell, int]] = {}
        self._corridor_interiors: dict[int, frozenset[Cell]] = {}

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.blocked

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Deterministic east, south, west, north neighbors after static blocks."""
        x, y = cell
        return [
            n
            for n in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
            if self.passable(n) and _edge(cell, n) not in self.blocked_edges
        ]

    def set_blocked(self, cell: Cell, blocked: bool = True) -> None:
        if not self.in_bounds(cell):
            raise ValueError("Cell outside grid")
        if blocked:
            self.blocked.add(cell)
        else:
            self.blocked.discard(cell)
        self._distance_cache.clear()
        self._corridors = None

    def set_edge_blocked(self, a: Cell, b: Cell, blocked: bool = True) -> None:
        if not _adjacent(a, b) or not self.in_bounds(a) or not self.in_bounds(b):
            raise ValueError("Edge must join adjacent in-bounds cells")
        if blocked:
            self.blocked_edges.add(_edge(a, b))
        else:
            self.blocked_edges.discard(_edge(a, b))
        self._distance_cache.clear()
        self._corridors = None

    @property
    def corridors(self) -> tuple[NarrowCorridor, ...]:
        """Detect maximal narrow chains, including corners, once per topology.

        A corridor interior consists of connected cells with exactly two free
        neighbors. Junctions/bays and dead ends are endpoints, not locked cells.
        Pure loops, or loops returning to the same junction, lack two distinct
        entrances and are excluded. Rebuild costs O(cells + edges).
        """
        if self._corridors is not None:
            return self._corridors
        neighbors = {
            (x, y): self.neighbors((x, y))
            for x in range(self.width)
            for y in range(self.height)
            if self.passable((x, y))
        }
        unvisited = {cell for cell, adjacent in neighbors.items() if len(adjacent) == 2}
        narrow = set(unvisited)
        corridors: list[NarrowCorridor] = []
        for seed in neighbors:
            if seed not in unvisited:
                continue
            component = {seed}
            queue = [seed]
            unvisited.remove(seed)
            while queue:
                cell = queue.pop()
                for adjacent in neighbors[cell]:
                    if adjacent in unvisited:
                        unvisited.remove(adjacent)
                        component.add(adjacent)
                        queue.append(adjacent)
            boundaries = sorted(
                {
                    adjacent
                    for cell in component
                    for adjacent in neighbors[cell]
                    if adjacent not in narrow
                }
            )
            if len(boundaries) != 2:
                continue
            first, last = boundaries
            following = [cell for cell in neighbors[first] if cell in component]
            if len(following) != 1:
                continue
            previous, cell = first, following[0]
            ordered = []
            while cell != last:
                ordered.append(cell)
                onward = [adjacent for adjacent in neighbors[cell] if adjacent != previous]
                if len(onward) != 1:
                    break
                previous, cell = cell, onward[0]
            if cell == last and len(ordered) == len(component):
                corridors.append(NarrowCorridor(len(corridors), (first, last), tuple(ordered)))
        self._corridors = tuple(corridors)
        self._corridor_by_cell = {
            cell: corridor for corridor in corridors for cell in corridor.interior
        }
        self._corridor_indices = {
            corridor.id: {cell: index for index, cell in enumerate(corridor.cells)}
            for corridor in corridors
        }
        self._corridor_interiors = {
            corridor.id: frozenset(corridor.interior) for corridor in corridors
        }
        return self._corridors

    def corridor_move_allowed(
        self,
        robot: RobotID,
        origin: Cell,
        target: Cell,
        table: ReservationTable,
    ) -> bool:
        """Enforce one travel direction within each currently occupied corridor.

        Apply this query *before* committing every normal or alternative move.
        It supplements vertex/edge reservations; it does not reserve space.
        Pending traversal direction is inferred from ordered corridor cells.
        A stationary interior occupant has unknown direction, so new entrants
        wait until that robot commits its next edge; residents may still exit.
        Same-direction followers can enter with a physically free gap. Origins
        of exit moves continue holding the corridor until arrival. Resources
        release automatically when the interior is empty, with no stale locks.
        Query cost is O(robots); topology detection is cached.
        """
        if not _adjacent(origin, target):
            return False
        _ = self.corridors
        affected = {
            corridor.id: corridor
            for cell in (origin, target)
            if (corridor := self._corridor_by_cell.get(cell)) is not None
        }
        if not affected:
            return True
        positions, moves = table.positions, table.moves
        for corridor in affected.values():
            indices = self._corridor_indices[corridor.id]
            if origin not in indices or target not in indices:
                return False
            direction = 1 if indices[target] > indices[origin] else -1
            interior = self._corridor_interiors[corridor.id]
            entering = origin not in interior and target in interior
            for other, position in positions.items():
                if other == robot:
                    continue
                move = moves.get(other)
                held = {position} if move is None else {position, move.target}
                if not held.intersection(interior):
                    continue
                if move is None or move.origin not in indices or move.target not in indices:
                    if entering:
                        return False
                    continue
                other_direction = 1 if indices[move.target] > indices[move.origin] else -1
                if other_direction != direction:
                    return False
        return True

    def distance_map(self, goal: Cell) -> dict[Cell, int]:
        """Reverse BFS distances; callers must treat the returned cache as read-only."""
        if not self.passable(goal):
            return {}
        if goal in self._distance_cache:
            self._distance_cache.move_to_end(goal)
            return self._distance_cache[goal]
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for neighbor in self.neighbors(cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)
        self._distance_cache[goal] = distances
        if len(self._distance_cache) > self.cache_size:
            self._distance_cache.popitem(last=False)
        return distances

    def distance(self, start: Cell, goal: Cell) -> int | None:
        return self.distance_map(goal).get(start)

    def shortest_path(
        self,
        start: Cell,
        goal: Cell,
        blocked_extra: Iterable[Cell] = (),
        blocked_edges: Iterable[Edge] = (),
    ) -> list[Cell] | None:
        """Exact shortest static/detour route including start and destination."""
        extra = set(blocked_extra) - {start}
        edges = {_edge(a, b) for a, b in blocked_edges}
        if not self.passable(start) or not self.passable(goal) or goal in extra:
            return None
        distances = self.distance_map(goal)
        if start not in distances:
            return None
        if not extra and not edges:
            # The reverse BFS is an exact heuristic, so greedy distance descent
            # is an exact shortest path and avoids thousands of repeated A*s.
            path = [start]
            while path[-1] != goal:
                path.append(
                    next(
                        n
                        for n in self.neighbors(path[-1])
                        if distances.get(n) == distances[path[-1]] - 1
                    )
                )
            return path
        serial = count()
        frontier = [(distances[start], 0, next(serial), start)]
        costs = {start: 0}
        came_from: dict[Cell, Cell] = {}
        while frontier:
            _, cost, _, cell = heapq.heappop(frontier)
            if cost != costs.get(cell):
                continue
            if cell == goal:
                return self._reconstruct(came_from, cell)
            for neighbor in self.neighbors(cell):
                if neighbor in extra or _edge(cell, neighbor) in edges:
                    continue
                new_cost = cost + 1
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    came_from[neighbor] = cell
                    heapq.heappush(
                        frontier,
                        (
                            new_cost + distances.get(neighbor, self.width * self.height),
                            new_cost,
                            next(serial),
                            neighbor,
                        ),
                    )
        return None

    @staticmethod
    def _reconstruct(came_from: dict, current: object) -> list:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))

    def timed_path(
        self,
        start: Cell,
        goal: Cell,
        start_tick: int,
        step_ticks: int,
        reservations: ReservationTable,
        robot: RobotID,
        horizon_ticks: int = 120,
        max_expansions: int = 12_000,
        blocked_extra: Iterable[Cell] = (),
    ) -> list[tuple[int, Cell]] | None:
        """Find a conservative space-time A* prefix, with one-tick waits.

        The default horizon is 60 seconds at 0.5-second simulator ticks. A
        horizon/expansion-limited result can end before the goal; callers use its
        next leg and must recheck it with ``try_reserve_move`` on execution.
        Returns None if static disconnection or no spatial progress is possible.
        Paths contain (arrival_tick, cell), including the initial state.
        ``blocked_extra`` forbids temporary transit cells (such as unrelated
        charger ports); the current start cell is always permitted for egress.
        """
        if step_ticks <= 0 or horizon_ticks <= 0 or max_expansions <= 0:
            raise ValueError("Step, horizon, and expansion limits must be positive")
        if start_tick < reservations.tick:
            raise ValueError("Planning cannot begin before reservation clock")
        extra = set(blocked_extra) - {start}
        if goal in extra:
            return None
        distances = self.distance_map(goal)
        if start not in distances:
            return None
        initial = (start_tick, start)
        if start == goal:
            return [initial]
        end_tick = start_tick + horizon_ticks
        serial = count()
        frontier = [(distances[start] * step_ticks, distances[start], next(serial), initial)]
        visited = {initial}
        came_from: dict[tuple[int, Cell], tuple[int, Cell]] = {}
        # Occupancy only becomes freer until the final already committed
        # arrival. Beyond it, later arrival at the same cell cannot improve a
        # route. Dominance pruning avoids horizon x grid exploration when a
        # stationary robot blocks the goal.
        last_release = max(
            (move.end_tick for move in reservations.moves.values()), default=start_tick
        )
        stable_arrivals: dict[Cell, int] = {}
        if start_tick >= last_release:
            stable_arrivals[start] = start_tick
        best = initial
        expansions = 0
        while frontier and expansions < max_expansions:
            _, _, _, state = heapq.heappop(frontier)
            tick, cell = state
            expansions += 1
            if (distances.get(cell, float("inf")), tick) < (distances[best[1]], best[0]):
                best = state
            if cell == goal:
                return self._reconstruct(came_from, state)
            for neighbor in [*self.neighbors(cell), cell]:
                if neighbor in extra:
                    continue
                if neighbor == cell and tick >= last_release:
                    continue
                duration = 1 if neighbor == cell else step_ticks
                arrival = tick + duration
                next_state = (arrival, neighbor)
                if arrival > end_tick or next_state in visited:
                    continue
                if arrival >= last_release and arrival >= stable_arrivals.get(
                    neighbor, float("inf")
                ):
                    continue
                if not reservations.can_traverse(robot, cell, neighbor, tick, duration):
                    continue
                if arrival >= last_release:
                    stable_arrivals[neighbor] = arrival
                visited.add(next_state)
                came_from[next_state] = state
                remaining = distances.get(neighbor, self.width * self.height)
                heapq.heappush(
                    frontier,
                    (
                        arrival - start_tick + remaining * step_ticks,
                        remaining,
                        next(serial),
                        next_state,
                    ),
                )
        if best[1] == start:
            return None
        return self._reconstruct(came_from, best)


__all__ = ["Cell", "Edge", "GridPlanner", "MoveReservation", "NarrowCorridor", "ReservationTable"]
