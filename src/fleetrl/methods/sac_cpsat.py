"""SAC implementation for sac_cpsat; common environment and entity encoder."""

from stable_baselines3 import SAC


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    pk["net_arch"] = [128, 128]
    return SAC(
        "MultiInputPolicy",
        env,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        seed=tc.seed,
        device=tc.device,
        verbose=0,
        policy_kwargs=pk,
        buffer_size=tc.buffer_size,
        learning_starts=tc.learning_starts,
        batch_size=tc.batch_size,
        train_freq=tc.train_freq,
        gradient_steps=tc.gradient_steps,
        tau=tc.tau,
        ent_coef="auto",
    )


def load(path, **kwargs):
    return SAC.load(path, **kwargs)
