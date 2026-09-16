"""Priority/age/nearest feasible dispatch with threshold charging."""

from ..optimizer import FleetOptimizer


def controller(config):
    return FleetOptimizer(config.copy(controller="heuristic", advanced_charging=False))
