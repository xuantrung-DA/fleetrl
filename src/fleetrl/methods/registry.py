"""Stable study identities. Each implementation is imported only when selected."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class MethodSpec:
    variant_id: str
    name: str
    algorithm: str | None
    backend: str = "cpsat"
    action_mode: str = "discrete"
    controller: str = "hybrid"
    advanced_charging: bool = True
    comparison: str = "learning"

    @property
    def learnable(self):
        return self.algorithm is not None

    def module(self):
        return import_module(f"fleetrl.methods.{self.name}")

    def configure(self, config):
        return config.copy(
            method=self.name,
            backend=self.backend,
            action_mode=self.action_mode,
            controller=self.controller,
            advanced_charging=self.advanced_charging,
        )


METHODS = {
    s.name: s
    for s in [
        MethodSpec(
            "V01",
            "heuristic",
            None,
            controller="heuristic",
            advanced_charging=False,
            comparison="baseline",
        ),
        MethodSpec("V02", "fixed_cpsat", None, controller="fixed", comparison="baseline"),
        MethodSpec("V03", "qlearning_cpsat", "qlearning", comparison="representation"),
        MethodSpec("V04", "dqn_cpsat", "dqn"),
        MethodSpec("V05", "a2c_cpsat", "a2c"),
        MethodSpec("V06", "ppo_cpsat", "ppo"),
        MethodSpec("V07", "qrdqn_cpsat", "qrdqn"),
        MethodSpec("V08", "recurrent_ppo_cpsat", "recurrent_ppo", comparison="memory"),
        MethodSpec(
            "V09", "ppo_threshold_cpsat", "ppo", advanced_charging=False, comparison="charging"
        ),
        MethodSpec(
            "V10", "sac_cpsat", "sac", action_mode="continuous", comparison="continuous_action"
        ),
        MethodSpec(
            "V11", "td3_cpsat", "td3", action_mode="continuous", comparison="continuous_action"
        ),
        MethodSpec(
            "V12",
            "mappo_dispatch",
            "mappo",
            backend="arbiter",
            action_mode="multiagent",
            comparison="dispatch",
        ),
        MethodSpec(
            "V13", "forecast_mpc_milp", None, backend="mpc", controller="fixed", comparison="system"
        ),
        MethodSpec("V14", "ppo_milp", "ppo", backend="milp", comparison="backend"),
    ]
}
ALIASES = {
    "B0": "heuristic",
    "B1": "fixed_cpsat",
    "H": "ppo_cpsat",
    "A": "ppo_threshold_cpsat",
    "fixed": "fixed_cpsat",
    "hybrid": "ppo_cpsat",
    "ablation": "ppo_threshold_cpsat",
}
ALIASES.update({s.variant_id: s.name for s in METHODS.values()})


def get_method(name):
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}; choose {list(METHODS)}")
    return METHODS[name]
