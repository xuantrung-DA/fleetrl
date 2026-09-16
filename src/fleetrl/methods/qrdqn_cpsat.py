"""QRDQN implementation for qrdqn_cpsat; common environment and entity encoder."""

from sb3_contrib import QRDQN


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    pk.update(net_arch=[128, 128], n_quantiles=50)
    return QRDQN(
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
    return QRDQN.load(path, **kwargs)
