"""PPO retrained with threshold charging; independent checkpoint identity."""

from .ppo_cpsat import load as load


def build(env, tc):
    from .ppo_cpsat import build as build_ppo

    if env.get_attr("config")[0].advanced_charging:
        raise ValueError("charging ablation requires advanced_charging=False")
    return build_ppo(env, tc)
