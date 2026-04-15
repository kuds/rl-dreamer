# Reading DreamerV3 TensorBoard Logs

Every example script writes TensorBoard events to its `logdir`. Launch
TensorBoard with:

```bash
tensorboard --logdir ~/logdir
```

This doc explains what each of the main DreamerV3 metrics means, what a
healthy run looks like, and what failure modes to watch for.

Metric names shown here follow the upstream DreamerV3 naming convention;
minor renames happen between releases, so use this as a field guide
rather than a rigid spec.

---

## The metrics that actually matter

### Episode metrics

| Tag                         | Meaning                                                      |
|-----------------------------|--------------------------------------------------------------|
| `episode/score`             | Undiscounted return of the most recent completed episode    |
| `episode/length`            | Number of env steps in the most recent episode              |
| `episode/reward_rate`       | Reward per env step (useful when episode lengths vary)      |

What to expect:

- `episode/score` should trend **upward** over training. Noise is normal.
- On dense-reward tasks (DMC, Atari) you should see movement within a few
  hundred thousand env steps. On sparse-reward tasks (Crafter, Minecraft)
  you may see nothing for millions of steps — that is expected, not a bug.

### World model losses

| Tag                  | Meaning                                                         |
|----------------------|-----------------------------------------------------------------|
| `train/loss_dyn`     | KL between RSSM prior and stop-gradient posterior (dynamics)    |
| `train/loss_rep`     | KL between posterior and stop-gradient prior (representation)  |
| `train/loss_dec`     | Decoder / observation reconstruction loss                      |
| `train/loss_reward`  | Reward head prediction loss                                    |
| `train/loss_cont`    | Continue (1 - done) prediction loss                            |
| `train/loss_model`   | Total world model loss                                         |

Healthy trajectories:

- `loss_dec` should drop sharply in the first few thousand steps as the
  decoder locks onto the observation distribution, then plateau.
- `loss_dyn` and `loss_rep` should both be above the `kl.free` free-bits
  floor (default 1.0 nats) and should *not* collapse to zero.
- `loss_reward` should track how predictable rewards are — on very
  sparse-reward tasks it stays low simply because most predictions are
  zero. Use `train/reward_pred_mae` (if logged) as a sharper signal.

### Actor / critic losses

| Tag                      | Meaning                                                    |
|--------------------------|------------------------------------------------------------|
| `train/loss_actor`       | Negative imagined λ-return plus entropy bonus              |
| `train/loss_critic`      | Regression of critic onto λ-returns                        |
| `train/actor_ent`        | Policy entropy (averaged across imagined rollouts)         |
| `train/actor_logprob`    | Mean log-probability of actions taken in imagination       |
| `train/imag_return`      | Mean imagined λ-return per rollout                         |
| `train/imag_value`       | Mean critic value on imagined states                       |

Healthy trajectories:

- `imag_return` should rise alongside `episode/score`. If imagined return
  climbs but real return does not, your world model is over-optimistic —
  see the failure-mode list below.
- `actor_ent` should decrease slowly. A sudden crash in entropy usually
  means the policy has collapsed onto a dominant action.
- `loss_critic` should decrease and then oscillate at low magnitude.
  Rising critic loss late in training means the targets are drifting.

### Throughput / hardware

| Tag                   | Meaning                                                       |
|-----------------------|---------------------------------------------------------------|
| `timer/agent_step`    | ms per agent step (policy + env interaction)                  |
| `timer/train_step`    | ms per learner gradient update                                |
| `timer/sample_step`   | ms per replay sample                                          |
| `counters/step`       | Total env steps                                               |
| `counters/update`     | Total gradient updates                                        |

Use these to diagnose "training feels slow" complaints. If
`timer/train_step` dominates, you're GPU-bound. If `timer/agent_step`
dominates, your env is the bottleneck — parallelize it.

---

## What a healthy run looks like

Rough shape, not absolute numbers:

```
episode/score       ____.----"""""    (noisy but rising)
train/loss_dec      \_________          (drops fast, then flat)
train/loss_dyn      \____----____       (drops to ~free bits, then flat)
train/loss_rep      \____----____
train/loss_actor    ~~\_____~~~__       (noisy, trending down)
train/actor_ent     ----\_________      (slow steady decrease)
train/imag_return       ___.---""       (rises with episode/score)
```

---

## Common failure modes

**1. Episode score is flat and `loss_dec` never drops.**
The world model hasn't learned to reconstruct observations at all. Usually
a config problem — the wrong `enc.simple.cnn_keys` / `mlp_keys` regex,
or an observation that isn't shaped the way DreamerV3 expects. Print
`env.obs_space` at startup and confirm the key you encode actually exists.

**2. `loss_dyn` collapses to zero.**
The prior has matched the posterior too aggressively — latent states
carry no information. Usually means the KL free bits floor is too low or
reconstruction loss is too weak. Revert to the defaults.

**3. `imag_return` shoots up but `episode/score` doesn't.**
The world model is hallucinating rewards. Causes:
- Reward head is overfitting to rare positive transitions. Try a
  smaller replay buffer or more reward samples.
- Imagination horizon is too long for the model's reliable horizon.
  Reduce `imag_horizon`.
- Continue head is underestimating episode termination, so imagined
  rollouts pretend to live forever. Check `train/loss_cont`.

**4. `actor_ent` crashes early.**
Policy collapse. Increase `actor.entropy` or reduce `run.train_ratio`
so you get more env diversity per update.

**5. `loss_critic` diverges late.**
Value targets are drifting. Make sure your critic EMA / target network
is configured — the defaults handle this, so if you have edited them,
revert.

**6. "train_step" latency climbs over time.**
Replay buffer on a slow disk, or JAX recompiling because shapes shift.
Point `--logdir` at a local SSD and make sure batch/length are fixed.

---

## Where to log from

The examples in this repo use the standard DreamerV3 logger stack:

```python
logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),
    embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    embodied.logger.TensorBoardOutput(logdir),
])
```

- **TerminalOutput** prints a subset of metrics to stdout so you can see
  progress without a browser.
- **JSONLOutput** writes `metrics.jsonl`, which is easy to ingest with
  `pandas.read_json(..., lines=True)` if you want to make your own plots.
- **TensorBoardOutput** writes the `events.out.tfevents.*` file that
  TensorBoard reads.

If you want to also push to Weights & Biases, add
`embodied.logger.WandBOutput(...)` to the list and make sure
`pip install wandb` is available.
