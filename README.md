# rl-dreamer

Working examples of using [DreamerV3](https://github.com/danijar/dreamerv3) — a
world-model based reinforcement learning agent that masters a wide range of
tasks with a single set of hyperparameters.

DreamerV3 learns a latent world model from experience and uses it to imagine
rollouts for policy optimization. Key properties:

- **Single fixed set of hyperparameters** across domains (continuous/discrete,
  pixels/vectors, sparse/dense rewards).
- **Scales with model size** — larger agents are more sample-efficient.
- **Robust** to reward scale via symlog predictions and two-hot encoding.

This repository contains ready-to-run scripts showing how to train DreamerV3 on
a variety of standard benchmarks, plus a template for plugging in your own
environment.

## Repository layout

```
rl-dreamer/
├── README.md
├── requirements.txt
├── .gitignore
├── examples/
│   ├── README.md
│   ├── 01_gym_cartpole.py        # Classic control (vector obs, discrete acts)
│   ├── 02_dmc_walker.py          # DeepMind Control Suite (pixel obs, continuous acts)
│   ├── 03_atari_pong.py          # Atari (pixel obs, discrete acts)
│   ├── 04_crafter.py             # Crafter (pixel obs, discrete acts, sparse rewards)
│   ├── 05_minigrid.py            # MiniGrid (pixel obs, discrete acts)
│   ├── 06_minecraft.py           # Minecraft via MineRL (paper's flagship task)
│   └── 07_custom_env.py          # Template for your own Gym-compatible env
├── notebooks/
│   └── dreamerv3_colab.ipynb     # Runnable Google Colab walkthrough
├── docs/
│   ├── architecture.md           # World-model + actor-critic diagrams
│   ├── LEARNING.md               # Curated reading / watching list
│   └── tensorboard_guide.md      # What each training metric means
└── scripts/
    ├── train.py                  # Generic CLI trainer
    ├── evaluate.py               # Load and roll out a trained agent
    ├── record.py                 # Save MP4 videos of a trained agent
    ├── visualize_dreams.py       # World-model reconstruction & dream videos
    └── visualize_network.py      # Render architecture diagrams as PNGs
```

## Run it on Google Colab

No GPU at hand? `notebooks/dreamerv3_colab.ipynb` is a ready-to-run Colab
notebook that installs DreamerV3, clones this repo, and walks through
CartPole → DMC → Atari → Crafter → Minecraft. Open it directly on Colab:

<https://colab.research.google.com/github/kuds/rl-dreamer/blob/main/notebooks/dreamerv3_colab.ipynb>

Remember to switch the runtime to GPU (`Runtime → Change runtime type → GPU`)
before running the cells.

## Installation

DreamerV3 requires Python 3.10+ and JAX. GPU support is strongly recommended
for pixel-based environments.

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install core deps (CPU JAX by default)
pip install -r requirements.txt

# 3. For CUDA 12 GPUs, replace JAX with the CUDA build
pip install -U "jax[cuda12]"
```

### Environment-specific extras

Some examples pull in additional environment suites. Install only what you
need:

```bash
# DeepMind Control Suite (example 02)
pip install dm_control

# Atari (example 03)
pip install "gymnasium[atari,accept-rom-license]" ale-py

# Crafter (example 04)
pip install crafter

# MiniGrid (example 05)
pip install minigrid

# Minecraft via MineRL (example 06)
#   Also needs Java 8 and (on headless machines) xvfb.
#   See https://minerl.readthedocs.io for OS-specific setup.
pip install "minerl==1.0.2"
```

## Quick start

Train DreamerV3 on CartPole in under a minute on CPU:

```bash
python examples/01_gym_cartpole.py --logdir ~/logdir/cartpole
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ~/logdir
```

## Using the generic trainer

`scripts/train.py` wraps the common setup (logger, replay, env stack, agent)
and accepts a `--task` flag along with any DreamerV3 config overrides:

```bash
python scripts/train.py \
    --task gym_CartPole-v1 \
    --preset size12m \
    --logdir ~/logdir/cartpole \
    --run.steps 50000
```

Supported task prefixes:

| Prefix     | Example                       | Notes                           |
|------------|-------------------------------|---------------------------------|
| `gym_`     | `gym_CartPole-v1`             | Any Gymnasium environment       |
| `dmc_`     | `dmc_walker_walk`             | DeepMind Control Suite          |
| `atari_`   | `atari_pong`                  | Atari via ALE                   |
| `crafter_` | `crafter_reward`              | Crafter (reward / noreward)     |
| `minigrid_`| `minigrid_MiniGrid-Empty-5x5` | MiniGrid                        |
| `minecraft_`| `minecraft_diamond`          | Real Minecraft via MineRL       |

## Model size presets

DreamerV3 ships with a family of sizes. Smaller = faster, larger = more
sample-efficient:

| Preset    | Params | Typical use                      |
|-----------|--------|----------------------------------|
| `size1m`  | 1 M    | Debugging, CPU-only quick checks |
| `size12m` | 12 M   | Small/medium benchmarks          |
| `size25m` | 25 M   | Standard DMC / Atari100k         |
| `size50m` | 50 M   | Harder visual tasks              |
| `size100m`| 100 M  | Crafter, large-scale runs        |
| `size200m`| 200 M  | Frontier experiments             |

Pass the preset to any example via `--preset sizeNm`.

## Recording rollout videos

Once you have a trained checkpoint, `scripts/record.py` writes one MP4
per episode to `<logdir>/videos/`:

```bash
pip install imageio imageio-ffmpeg

python scripts/record.py \
    --task gym_CartPole-v1 \
    --preset size1m \
    --logdir ~/logdir/cartpole \
    --episodes 3
```

Use the same `--task` and `--preset` you used with `scripts/train.py`.
The script works for every task suite listed above — for pixel-obs
envs (DMC, Atari, Crafter, MiniGrid, Minecraft) frames come directly
from the observation stream, and for Gym classic-control envs the
underlying env's `render(mode='rgb_array')` is used.

## Visualizing world-model dreams

`scripts/visualize_dreams.py` shows what the world model *thinks* and
*imagines*. Unlike `record.py` (which captures the agent's behaviour in
the real environment), this script uses `agent.report()` to extract:

- **Reconstructions** — the decoder's output when fed posterior latents
  from real observations.
- **Open-loop imagination** — what the model "dreams" when running
  purely from the prior, with no new observations.

```bash
pip install imageio imageio-ffmpeg

python scripts/visualize_dreams.py \
    --task crafter_reward \
    --preset size25m \
    --logdir ~/logdir/crafter
```

Outputs are written to `<logdir>/dreams/` (override with `--output`).
Each visual key from `agent.report()` is saved as both an MP4 and a
contact-sheet PNG (a grid of sampled frames).

The script first tries to load a batch from the saved replay buffer
under `<logdir>/replay/`. If none exists it falls back to collecting a
live rollout. A trained checkpoint is required.

## Visualizing the network

`scripts/visualize_network.py` renders schematic diagrams of the
DreamerV3 architecture as PNGs you can embed in slides, papers, or
notebooks. It does not require a trained checkpoint — the diagrams are
statically drawn with matplotlib — so it's useful as a teaching aid
alongside [`docs/architecture.md`](docs/architecture.md).

```bash
pip install matplotlib

# Render all three diagrams into docs/figures/
python scripts/visualize_network.py

# Or pick a different output directory
python scripts/visualize_network.py --output /tmp/dreamer_figures

# Render only one diagram
python scripts/visualize_network.py --only world_model
```

Three figures are produced:

| File                  | What it shows                                                      |
|-----------------------|--------------------------------------------------------------------|
| `world_model.png`     | RSSM: encoder, GRU, prior/posterior, decoder and reward/continue heads |
| `imagination.png`     | Open-loop imagination rollout scored by actor and critic           |
| `pipeline.png`        | Three nested loops: environment → replay → world model → actor-critic |

## Learning resources

If you are new to DreamerV3 or world-model-based RL in general, start
with these docs:

- [`docs/architecture.md`](docs/architecture.md) — diagrams of the
  world model, imagination loop, and how data flows through the agent.
- [`docs/LEARNING.md`](docs/LEARNING.md) — curated reading / watching
  list, including the paper, earlier Dreamer iterations, related
  benchmarks, and external talks and videos.
- [`docs/tensorboard_guide.md`](docs/tensorboard_guide.md) — what each
  training metric means, what a healthy run looks like, and common
  failure modes.

## References

- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3),
  2023. <https://arxiv.org/abs/2301.04104>
- Official implementation: <https://github.com/danijar/dreamerv3>

## License

The examples in this repository are released under the MIT License. DreamerV3
itself is licensed by its authors — see the upstream repository for details.
