# RL architecture and experimental contract

The policy is one centralized PPO agent that chooses six solver cost profiles every five simulated seconds. Robots do not own learned policies. CP-SAT determines assignments and charging schedules; a separate planner and simulator enforce feasibility. This preserves the proposal's central research question: does state-dependent preference selection improve joint task and charging coordination over the same optimizer with fixed preferences?

## Why this implementation

PPO uses clipped policy updates with Generalized Advantage Estimation. Stable-Baselines3 supplies the tested rollout buffer, categorical action distribution, time-limit handling, optimizer state serialization, and update implementation. Reimplementing those mechanisms would increase defect risk without strengthening this project's scientific contribution. PPO's clipping is not a safety guarantee and does not guarantee monotonic improvement. A KL threshold and gradient clipping provide additional optimization controls, while physical constraints remain outside RL. [PPO paper](https://arxiv.org/abs/1707.06347), [GAE paper](https://arxiv.org/abs/1506.02438), [SB3 PPO reference](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

The six actions are balanced, lateness priority, energy reserve, service availability, reduced empty travel, and regional balance. Every profile is legal. An action mask would incorrectly conflate profile validity with individual robot-task feasibility, so no action masking library is used. Padding masks belong to observation processing only. A profile can result in HOLD when physical actions are infeasible.

## Two comparable policy encoders

**`mlp` is the proposal reference.** Flatten all scaled observations and presence masks, then use independent actor and critic MLPs with two 128-unit tanh layers. Keep it in every architectural ablation.

**`entities` is the default, deliberately modest extension.** Each robot, task, and charger-port row passes through a shared two-layer 64-unit encoder for its entity type. Presence masks exclude padded rows. For each type, masked mean, masked maximum, and learned global-query attention pool the rows. A separate global-context encoder includes fleet aggregates and demand history. Concatenation followed by a 256-unit fusion layer feeds independent actor and critic MLPs, each with two 128-unit tanh layers. The actor produces six categorical logits; the critic produces one scalar state value. There is no dropout, no recurrent hidden state, no Transformer stack, and no per-robot action head.

Shared row encoders make representation independent of arbitrary robot/task row ordering, provided features and masks are permuted together. Mean and maximum summarize common and exceptional states; attention can emphasize urgent tasks or low-battery robots using global conditions. Empty entity sets have an exactly zero pooled representation. This is an implementation choice inspired by permutation-invariant set modeling, not a claimed reproduction of a new fleet-RL paper or a proven performance improvement. [Deep Sets](https://arxiv.org/abs/1703.06114), [SB3 custom feature extractors](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).

The encoder cannot reconstruct detailed spatial relationships that were omitted from observations. It also cannot observe future event tapes. History windows reduce uncertainty about demand without making a partially observed system fully observable. Recurrent PPO and graph message passing should be considered only after ablations demonstrate a concrete history or interaction bottleneck.

## PPO settings and credit assignment

The proposal defaults remain learning rate 0.0003, gamma 0.995, GAE lambda 0.95, clip 0.2, rollout 1024 steps per environment, and minibatch 256. Additional explicit settings include entropy regularization, value-loss coefficient, maximum gradient norm, target KL, and update epochs. The effective discounted horizon is about 200 decisions, or 1000 simulated seconds; GAE's practical decay is shorter. Sparse delayed charging benefits and terminal backlog penalties can therefore remain difficult even with this network. Inspect value explained variance and reward components before assuming the network needs to grow.

Inputs are scaled by fixed physical constants inside the environment. Presence masks stay binary. There is no learned observation or reward normalizer, so inference never depends on an omitted VecNormalize pickle. PPO normalizes advantages within updates. The reward follows the proposal: completed tasks, backlog, priority-weighted overdue backlog, energy, safety events, and a terminal penalty for unfinished work and accumulated lateness. Different reward weights are a new experiment and require validation, not a silent post-hoc change.

## Data separation, checkpoints, and continuation

The default training seed range is 0–199; validation defaults to 1000, 1001 and 1002. The benchmark owns the separate held-out test seed range. Configuration rejects train/validation seed overlap. Training generates fresh episodes; repeated validation uses the same prescribed seeds and a fixed target configuration. Validation ranks checkpoints lexicographically: fewer collisions, edge conflicts, energy emergencies and reserve violations; then more completed tasks; then less total lateness; then fewer pending tasks. This fixed rule prioritizes feasible service, and all ties retain the earlier checkpoint. Validation always uses the exact base target configuration rather than the sampled training mixture. Only validation selects a best checkpoint. Held-out test results do not feed checkpoint selection or tuning.

A checkpoint contains policy, value network, optimizer state, and PPO counters. A resumed run restarts simulation episodes from a reproducible seed stream and continues learned weights and optimizer state; it does not claim bit-identical continuation of simulator, rollout buffer, Python RNG, or OS scheduling. Resume validation first evaluates the loaded policy, preventing a newly initialized callback from replacing a better resumed checkpoint with an unmeasured one. Evaluation timing includes policy inference plus the downstream decision stack. Training validates the initial policy too, so a weak run can legitimately keep an untrained policy as its best checkpoint. Training writes both `best_model.zip` selected on validation and `final_model.zip` for the last update.

## Curriculum and interaction budget

For budgets of at least three rollouts, the optional curriculum allocates roughly 15% to at most five robots with lighter demand and no disturbances or burst, 25% to at most ten robots with intermediate demand and no disturbances, and 60% to a target mixture. For reference fleets of at least ten robots, this mixture crosses robot counts 10/15/20 with per-robot demand multipliers 0.7/1.0/1.3; both lists are explicit configuration fields. Each reset samples one variant through its own reproducible RNG, independently of the training scenario seed stream. The sampled demand is base demand × sampled robot count / base robot count × multiplier. Episode horizon and padded observation spaces stay unchanged. Demand scales with fleet size so a smaller warmup fleet is not inadvertently overloaded. Each stage starts fresh episodes and keeps the same padded observation dimensions. PPO finishes full rollouts; reported actual steps can exceed the requested budget. No warmup or intermediate configuration replaces target validation.

A resumed run skips the warmup and trains on the target distribution, including this episode mixture when curriculum remains enabled. Reference fleets below ten robots and runs with curriculum disabled keep the exact configured fleet. Monitor CSVs record every completed training episode’s seed, actual robot count and demand rate; stage manifests record the complete available mixture. Training with `controller: fixed` or `heuristic` is rejected because those controllers would remove the policy's action influence. The default CPU policy is small; spawned CPU workers parallelize simulation when `n_envs > 1`. CUDA is supported through SB3/PyTorch but must be measured on the user's hardware; a GPU cannot eliminate the CPU simulator and CP-SAT cost.

## What counts as evidence

A smoke run proves that observations reach the network, finite rollout/update steps execute, checkpoints save/load, deterministic predictions survive reload, and short evaluation terminates. It does not prove that the agent learned a useful fleet policy. A credible result needs independent training seeds, paired scenario tapes, full-shift evaluation, constraint counts, unfinished and late tasks, energy/throughput, confidence intervals, and identical candidate generation/solver budgets for learned and fixed profiles. [SB3 experimental guidance](https://stable-baselines3.readthedocs.io/en/v2.4.1/guide/rl_tips.html), [statistical evaluation of deep RL](https://arxiv.org/abs/2108.13264).

## Debugging a poor run

1. Check invariant failures, action feasibility and solver status first. A reward increase cannot repair a simulator defect.
2. Compare all six fixed profiles and a random-profile policy on the same tapes. If actions rarely change plans, policy learning has little controllable signal.
3. Inspect completed, pending, late, charge-wait, energy and safety components together. A scalar mean reward can hide starvation of heavy or urgent jobs.
4. Inspect action entropy, approximate KL, clip fraction, explained variance and finite loss values. Adjust one issue at a time, using validation only.
5. Compare `entities` against `mlp` under equal interaction budgets and report measured wall-clock cost separately.
6. Only after the above, vary demand, charging pressure, curriculum, seeds and budgets. Preserve the held-out map and test tapes until the experiment is frozen.

Sources were checked on 2026-09-11. Research supports the algorithmic mechanisms, not a guarantee of task-specific performance.
