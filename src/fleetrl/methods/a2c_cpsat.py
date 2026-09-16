"""A2C implementation for a2c_cpsat; common environment and entity encoder."""

from stable_baselines3 import A2C


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    return A2C(
        "MultiInputPolicy",
        env,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        seed=tc.seed,
        device=tc.device,
        verbose=0,
        policy_kwargs=pk,
        n_steps=tc.n_steps,
        gae_lambda=tc.gae_lambda,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        max_grad_norm=tc.max_grad_norm,
    )


def load(path, **kwargs):
    return A2C.load(path, **kwargs)
