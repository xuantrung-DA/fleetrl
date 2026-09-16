"""RecurrentPPO implementation for recurrent_ppo_cpsat; common environment and entity encoder."""

from sb3_contrib import RecurrentPPO


def build(env, tc):
    from ..learning.training import encoder_kwargs

    pk = encoder_kwargs(tc)
    pk.update(lstm_hidden_size=128, n_lstm_layers=1)
    return RecurrentPPO(
        "MultiInputLstmPolicy",
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
    return RecurrentPPO.load(path, **kwargs)
