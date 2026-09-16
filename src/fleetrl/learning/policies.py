"""Per-episode inference session: recurrent memory never leaks between tapes."""

import numpy as np


class InferenceSession:
    def __init__(self, policy):
        self.policy = policy
        self.state = None
        self.episode_start = np.array([True])
        self._fleetrl_checkpoint_path = getattr(policy, "_fleetrl_checkpoint_path", None)

    def reset(self):
        self.state = None
        self.episode_start = np.array([True])

    def predict(self, observation, deterministic=True):
        # SB3 base algorithms and custom policies all accept this signature.
        import inspect

        if "state" in inspect.signature(self.policy.predict).parameters:
            action, self.state = self.policy.predict(
                observation,
                state=self.state,
                episode_start=self.episode_start,
                deterministic=deterministic,
            )
        else:
            prediction = self.policy.predict(observation, deterministic=deterministic)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
        self.episode_start[:] = False
        return action, self.state
