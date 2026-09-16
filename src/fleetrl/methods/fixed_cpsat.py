"""Fixed objective profile, optimized charging and common CP-SAT scheduler."""

from ..optimizer import FleetOptimizer


def controller(config):
    return FleetOptimizer(config.copy(controller="fixed", advanced_charging=True))
