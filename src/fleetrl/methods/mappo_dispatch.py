"""Shared robot actor, centralized fleet critic and deterministic resource arbiter.

The policy samples per-robot proposals. PPO likelihoods refer to these sampled
proposals; arbitration is part of the transition, never substituted into the
likelihood. One transition and one shared reward are counted per fleet step.
"""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical

from ..observations import observation_space
from ..optimizer import earliest_slot, queue_fits
from ..types import Decision, DecisionPlan

MAX_CHOICES = 64
CANDIDATE_DIM = 12


def dispatch_space():
    values = dict(observation_space().spaces)
    values.update(
        candidates=spaces.Box(-5, 5, (20, MAX_CHOICES, CANDIDATE_DIM), dtype=np.float32),
        candidate_mask=spaces.Box(0, 1, (20, MAX_CHOICES), dtype=np.float32),
        active_agents=spaces.Box(0, 1, (20,), dtype=np.float32),
    )
    return spaces.Dict(values)


def dispatch_observation(owner, snapshot, obs):
    choices = {}
    candidates = owner.generate_candidates(snapshot)
    features = np.zeros((20, MAX_CHOICES, CANDIDATE_DIM), np.float32)
    mask = np.zeros((20, MAX_CHOICES), np.float32)
    mask[:, 0] = 1
    active = np.zeros(20, np.float32)
    for index, robot in enumerate(sorted(snapshot.robots, key=lambda r: r.id)):
        options = [c for c in candidates if c.decision.robot_id == robot.id]
        options.sort(
            key=lambda c: (
                c.decision.kind != "wait",
                c.decision.kind,
                c.decision.task_id or -1,
                c.decision.port_id or -1,
                c.decision.target_soc or 0,
                c.decision.target_position or (0, 0),
            )
        )
        if len(options) > MAX_CHOICES:
            raise ValueError("candidate count exceeds fixed MAPPO schema")
        choices[index] = (robot.id, options)
        active[index] = float(robot.status == "idle" and bool(options))
        for j, c in enumerate(options):
            d = c.decision
            kind = ["wait", "task", "charge", "reposition"].index(d.kind)
            features[index, j, kind] = 1
            features[index, j, 4:] = [
                c.duration_s / 900,
                c.energy_wh / 100,
                c.gain_wh / 100,
                c.tardiness_s / 600,
                c.zone / 3,
                float(c.available),
                d.target_soc or 0,
                d.metadata.get("empty_travel_m", d.metadata.get("travel_m", 0)) / 100,
            ]
            mask[index, j] = 1
    obs.update(candidates=np.clip(features, -5, 5), candidate_mask=mask, active_agents=active)
    return obs, choices


def arbitrate(owner, snapshot, choices, actions, deadline):
    occupied, queues = owner._calendars(snapshot)
    robots = {r.id: r for r in snapshot.robots}
    used_tasks = set()
    used_cells = set()
    decisions = []
    rejects = []
    cfg = owner.config
    for index, (rid, options) in sorted(
        choices.items(), key=lambda item: (robots[item[1][0]].soc, item[1][0])
    ):
        if not options:
            continue
        selected = int(actions[index])
        reason = None
        if not 0 <= selected < len(options):
            reason = "masked_action"
        elif perf_counter() >= deadline:
            reason = "arbiter_budget"
        else:
            c = options[selected]
            d = replace(c.decision, metadata=dict(c.decision.metadata))
            if d.kind == "task" and d.task_id in used_tasks:
                reason = "task_conflict"
            if d.kind == "reposition" and d.target_position in used_cells:
                reason = "parking_conflict"
            if d.kind == "charge":
                start = earliest_slot(
                    c.earliest_tick,
                    c.latest_tick,
                    c.duration_tick,
                    occupied[d.port_id],
                    owner._ticks(cfg.decision_s),
                )
                if start is None or not queue_fits(
                    owner._ticks(d.arrival_s),
                    start,
                    queues.get(c.station_id, []),
                    owner._queue_capacity(snapshot, c.station_id),
                ):
                    reason = "charging_conflict"
                else:
                    d.start_s = start * cfg.tick_s
                    d.end_s = (start + c.duration_tick) * cfg.tick_s
                    occupied[d.port_id].append((start, start + c.duration_tick))
                    queues.setdefault(c.station_id, []).append((owner._ticks(d.arrival_s), start))
        if reason:
            d = Decision(rid, "wait", reason=reason)
            rejects.append({"robot_id": rid, "reason": reason})
        else:
            if d.kind == "task":
                used_tasks.add(d.task_id)
            if d.kind == "reposition":
                used_cells.add(d.target_position)
        decisions.append(d)
    return DecisionPlan(
        snapshot.version,
        None,
        decisions,
        solver_status="ARBITRATED",
        candidate_count=sum(len(v[1]) for v in choices.values()),
        metadata={
            "backend": "arbiter",
            "solver_called": False,
            "arbitration_rejections": rejects,
            "priority": "ascending SOC then robot ID",
            "sampled_actions": np.asarray(actions).tolist(),
        },
    )


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(14 + CANDIDATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(40 + 14 + 10 + 6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        robot = obs["robots"].unsqueeze(2).expand(-1, -1, MAX_CHOICES, -1)
        logits = self.actor(torch.cat((robot, obs["candidates"]), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~obs["candidate_mask"].bool(), -1e9)
        pools = []
        for key, mask in [("robots", "robot_mask"), ("tasks", "task_mask"), ("ports", "port_mask")]:
            weight = obs[mask].unsqueeze(-1)
            pools.append((obs[key] * weight).sum(1) / weight.sum(1).clamp_min(1))
        value = self.critic(torch.cat([obs["global"]] + pools, dim=-1)).squeeze(-1)
        return Categorical(logits=logits), value


class MAPPO:
    def __init__(self, tc):
        self.tc = tc
        torch.manual_seed(tc.seed)
        self.device = torch.device(
            "cuda"
            if tc.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if tc.device == "auto"
            else tc.device
        )
        self.policy = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=tc.learning_rate, eps=1e-5)
        self.num_timesteps = 0
        self._n_updates = 0
        self.rng = np.random.default_rng(tc.seed)

    def tensors(self, obs, batch=False):
        return {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            if batch
            else torch.as_tensor(v, device=self.device, dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        with torch.no_grad():
            dist, _ = self.policy(self.tensors(observation))
            action = dist.logits.argmax(-1) if deterministic else dist.sample()
        return action[0].cpu().numpy(), None

    def learn_custom(self, env, steps, callback):
        obs, _ = env.reset()
        remaining = steps
        tc = self.tc
        while remaining > 0:
            observations = []
            actions = []
            logps = []
            values = []
            rewards = []
            dones = []
            for _ in range(min(tc.n_steps, remaining)):
                with torch.no_grad():
                    dist, value = self.policy(self.tensors(obs))
                    action = dist.sample()
                    logp = dist.log_prob(action)
                nxt, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                observations.append(obs)
                actions.append(action[0])
                logps.append(logp[0])
                values.append(value.item())
                rewards.append(reward)
                dones.append(terminated or truncated)
                self.num_timesteps += 1
                remaining -= 1
                obs = env.reset()[0] if dones[-1] else nxt
            with torch.no_grad():
                _, last = self.policy(self.tensors(obs))
            advantages = np.zeros(len(rewards), np.float32)
            gae = 0.0
            next_value = last.item()
            for i in reversed(range(len(rewards))):
                live = 1 - float(dones[i])
                delta = rewards[i] + tc.gamma * next_value * live - values[i]
                gae = delta + tc.gamma * tc.gae_lambda * live * gae
                advantages[i] = gae
                next_value = values[i]
            returns = torch.tensor(
                advantages + np.array(values), device=self.device, dtype=torch.float32
            )
            adv = torch.tensor(advantages, device=self.device)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            batch = self.tensors({k: np.stack([o[k] for o in observations]) for k in obs}, True)
            sampled = torch.stack(actions)
            old_logp = torch.stack(logps)
            for epoch in range(tc.n_epochs):
                stop = False
                for begin in range(0, len(rewards), tc.batch_size):
                    # Shuffle once per epoch, preserving aligned fleet-step tensors.
                    if begin == 0:
                        order = self.rng.permutation(len(rewards))
                    ids = torch.as_tensor(order[begin : begin + tc.batch_size], device=self.device)
                    mini = {k: v[ids] for k, v in batch.items()}
                    dist, value = self.policy(mini)
                    ratio = (dist.log_prob(sampled[ids]) - old_logp[ids]).exp()
                    active = mini["active_agents"]
                    denominator = active.sum().clamp_min(1)
                    unclipped = ratio * adv[ids, None]
                    clipped = ratio.clamp(1 - tc.clip_range, 1 + tc.clip_range) * adv[ids, None]
                    actor_loss = -(torch.minimum(unclipped, clipped) * active).sum() / denominator
                    entropy = (dist.entropy() * active).sum() / denominator
                    loss = (
                        actor_loss
                        + tc.vf_coef * (value - returns[ids]).square().mean()
                        - tc.ent_coef * entropy
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite MAPPO loss")
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), tc.max_grad_norm)
                    self.optimizer.step()
                    self._n_updates += 1
                    kl = (
                        (ratio - 1 - torch.log(ratio.clamp_min(1e-12))) * active
                    ).sum() / denominator
                    if tc.target_kl and kl.item() > 1.5 * tc.target_kl:
                        stop = True
                        break
                if stop:
                    break
            callback(self, info)
        return self

    def save(self, path):
        torch.save(
            {
                "schema": "mappo-v1",
                "config": asdict(self.tc),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.num_timesteps,
                "updates": self._n_updates,
                "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu", **kwargs):
        from ..config import TrainConfig

        data = torch.load(path, map_location=device, weights_only=False)
        if data["schema"] != "mappo-v1":
            raise ValueError("incompatible MAPPO schema")
        tc = TrainConfig(**data["config"])
        tc.device = device
        obj = cls(tc)
        obj.policy.load_state_dict(data["policy"])
        obj.optimizer.load_state_dict(data["optimizer"])
        obj.num_timesteps = data["steps"]
        obj._n_updates = data["updates"]
        obj.rng.bit_generator.state = data["rng"]
        torch.set_rng_state(data["torch_rng"].cpu())
        return obj


def build(env, tc):
    return MAPPO(tc)


def load(path, **kwargs):
    return MAPPO.load(path, **kwargs)
