# BrowserGym Single-Environment PPO

## Current State

Stage 1 verified the BrowserGym adapter path:

- `BrowserGymJAWSEnv` reset and step work.
- J.A.W.S raw observation keys are produced.
- `candidate_count=32`.
- Observation vector shape is `(653,)`.
- Action mask shape is `(224,)`.
- Action dimension is `224`.

## Stage 2 Goal

Stage 2 adds a single-environment PPO loop. It does not use `scenarios.json` steps,
does not add parallel workers, and does not modify deterministic detectors. The
policy chooses actions directly from the masked action space.

## Flow

`BrowserGymJAWSEnv`
-> `ObservationEncoder`
-> `PPOActorCritic`
-> `PPOAgent`
-> `AnomalyDetectionService`
-> `AutonomousRewardService`
-> PPO update

The environment reward is treated as temporary. Training recomputes reward from
observation changes, anomaly signals, optional known-bug matches, and repeat/no-op
penalties.

## Model

`PPOActorCritic` is a CPU-friendly MLP:

- Shared layers: `653 -> 256 -> 128 -> 64`
- Actor head: `64 -> 224`
- Critic head: `64 -> 1`

Invalid actions are masked by setting logits to `-1e9` before sampling or greedy
selection.

## Reward Signals

The autonomous reward combines:

- Anomaly reward for high, medium, and low confidence detections.
- Exploration reward for URL/state changes and new interactive clicks.
- Error signal reward for layout overlap, action errors, and broken navigation.
- Optional known bug match reward from `datasets/{site_id}/bug_catalog.json`.
- Penalties for repeated actions, repeated no-op, invalid/failed actions, and
  repeated no-change steps.

The final reward is clamped to `[-1.0, 1.5]`.

## Train

```powershell
python runners/train_browsergym_agent.py --site-id site001 --base-url http://localhost:9220 --episodes 5 --max-steps 10
```

Outputs:

- `artifacts/models/site001_browsergym_ppo.pt`
- `artifacts/browsergym/site001/rl_transition_log.jsonl`
- `artifacts/browsergym/site001/training_summary.json`
- `artifacts/browsergym/site001/detected_bugs.json`

## Evaluate

```powershell
python runners/evaluate_browsergym_agent.py --site-id site001 --base-url http://localhost:9220 --model-path artifacts/models/site001_browsergym_ppo.pt
```

Evaluation runs greedy policy actions for 1 to 3 episodes by default and prints
average reward, detected anomaly count, known bug matches, and simple
precision/recall-style metrics.
