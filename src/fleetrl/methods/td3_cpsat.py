"""TD3 implementation for td3_cpsat; common environment and entity encoder."""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    pk["net_arch"] = [128, 128]
    return TD3(
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
        action_noise=NormalActionNoise(np.zeros(5), 0.1 * np.ones(5)),
    )


def load(path, **kwargs):
    return TD3.load(path, **kwargs)
