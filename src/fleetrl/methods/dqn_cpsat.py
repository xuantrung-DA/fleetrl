"""DQN implementation for dqn_cpsat; common environment and entity encoder."""

from stable_baselines3 import DQN


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    pk["net_arch"] = [128, 128]
    return DQN(
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
        target_update_interval=tc.target_update_interval,
        exploration_fraction=tc.exploration_fraction,
        exploration_final_eps=tc.exploration_final_eps,
    )


def load(path, **kwargs):
    return DQN.load(path, **kwargs)
