# DreamerV3 Examples

Each script in this directory is a **standalone, runnable training loop** for
one environment. They all share the same structure:

1. Build a config by chaining `defaults` + a size preset + task overrides.
2. Construct a logger, replay buffer, and batched env.
3. Instantiate `dreamerv3.Agent` and call `embodied.run.train(...)`.

## Index

| File                       | Environment            | Obs     | Actions    | Needs GPU? |
|----------------------------|------------------------|---------|------------|------------|
| `01_gym_cartpole.py`       | Gymnasium CartPole     | vector  | discrete   | no         |
| `02_dmc_walker.py`         | DM Control (walker)    | pixel   | continuous | recommended|
| `03_atari_pong.py`         | Atari Pong             | pixel   | discrete   | recommended|
| `04_crafter.py`            | Crafter                | pixel   | discrete   | recommended|
| `05_minigrid.py`           | MiniGrid DoorKey       | pixel   | discrete   | optional   |
| `06_minecraft.py`          | Minecraft (MineRL)     | pixel+vec| hybrid    | required   |
| `07_custom_env.py`         | Your own env           | either  | either     | depends    |

## Running

All scripts accept the same DreamerV3 command-line flags, which override
anything set in the Python file. For instance:

```bash
python examples/01_gym_cartpole.py \
    --logdir ~/logdir/cartpole_debug \
    --run.steps 10000 \
    --batch_size 8
```

Common flags:

- `--logdir PATH` — where to save metrics, checkpoints, replay.
- `--run.steps N` — total environment steps to train for.
- `--run.train_ratio R` — gradient steps per env step * batch.
- `--run.log_every SECONDS` — logging cadence.
- `--jax.platform {cpu,gpu,tpu}` — force a JAX backend.

## Monitoring

All scripts write TensorBoard events to `<logdir>/`. Launch TensorBoard with:

```bash
tensorboard --logdir ~/logdir
```

Key metrics to watch:

- `train/return` — mean episode return on the training env.
- `train/length` — episode length.
- `train/loss_dyn`, `loss_rep`, `loss_dec` — world-model losses.
- `train/loss_actor`, `loss_critic` — policy losses.

## Tips

- **Start small.** Use `size1m` or `size12m` presets and a shrunken replay
  while you are getting your environment wired up. Only scale up once the
  plumbing is proven.
- **Watch the replay directory.** Replays can grow large quickly — set
  `--replay.size` low during debugging.
- **Pixel envs need CNN keys.** Make sure `enc.simple.cnn_keys` / `dec...`
  include the image stream name and `mlp_keys` exclude it (`$^`).
- **Dict observations.** Use a pipe-separated regex in `mlp_keys` to pick
  which scalar streams to encode — see `06_minecraft.py` for an example.
