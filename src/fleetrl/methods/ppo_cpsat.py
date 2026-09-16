"""PPO implementation for ppo_cpsat; common environment and entity encoder."""

from stable_baselines3 import PPO


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    return PPO(
        "MultiInputPolicy",
        env,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        seed=tc.seed,
        device=tc.device,
        verbose=0,
        policy_kwargs=pk,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        max_grad_norm=tc.max_grad_norm,
        target_kl=tc.target_kl,
    )


def load(path, **kwargs):
    return PPO.load(path, **kwargs)
