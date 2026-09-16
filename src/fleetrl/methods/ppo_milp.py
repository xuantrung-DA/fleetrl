"""PPO trained through SCIP MILP, sharing CP-SAT candidates and objective."""

from .ppo_cpsat import load as load


def build(env, tc):
    from .ppo_cpsat import build as build_ppo

    if env.get_attr("config")[0].backend != "milp":
        raise ValueError("PPO-MILP requires the MILP environment")
    return build_ppo(env, tc)
