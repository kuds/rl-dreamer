"""Train DreamerV3 on Minecraft — the diamond-collection flagship task.

DreamerV3 was the first published algorithm to collect diamonds in Minecraft
from scratch, without human data or a curriculum. This example reproduces
the same environment setup used in that result.

Two backends are supported:

1. **MineRL** (recommended, matches the paper):
       pip install minerl
   Requires Java 8 and a working OpenGL / xvfb stack. See the MineRL docs
   at https://minerl.readthedocs.io for platform-specific setup. On headless
   machines prefix the run with `xvfb-run -a`.

2. **Crafter** (lightweight, no Java):
       pip install crafter
   Not real Minecraft, but a 2D Minecraft-inspired benchmark that runs on
   any machine. Use this if MineRL is too heavy — see examples/04_crafter.py.

Run (GPU strongly recommended — the paper uses size200m):
    xvfb-run -a python examples/06_minecraft.py --logdir ~/logdir/minecraft

Notes:
    - `size25m` is used by default to keep memory/time manageable. For the
      full paper result, pass `--preset size200m` via scripts/train.py, or
      edit the preset below.
    - Episodes are long (24000 env steps). Training a strong diamond agent
      takes tens of millions of env steps.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def main():
    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import minecraft

    # ------------------------------------------------------------------
    # 1. Config: defaults + Minecraft preset.
    #
    # DreamerV3 ships with a built-in 'minecraft_diamond' config preset
    # that sets the appropriate action repeat, resolution, and reward
    # shaping for the MineRL diamond task.
    # ------------------------------------------------------------------
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["minecraft"])
    config = config.update(dreamerv3.Agent.configs["size25m"])
    config = config.update(
        {
            "logdir": "~/logdir/minecraft",
            "run.train_ratio": 16,
            "run.log_every": 300,
            "run.steps": 10_000_000,
            "batch_size": 16,
            "batch_length": 64,
            "enc.simple.cnn_keys": "image",
            "dec.simple.cnn_keys": "image",
            # Minecraft exposes a dict of inventory + equipment scalars in
            # addition to the image observation — encode them via the MLP.
            "enc.simple.mlp_keys": "inventory|inventory_max|equipped|health|hunger|breath|reward",
            "dec.simple.mlp_keys": "inventory|inventory_max|equipped|health|hunger|breath|reward",
        }
    )
    config = embodied.Flags(config).parse()

    def make_env(index: int):
        # The MinecraftDiamond env wraps MineRL's ObtainDiamondShovel task
        # with the observation and action shaping used in the paper.
        env = minecraft.MinecraftDiamond(
            repeat=1,
            size=(64, 64),
            break_speed=100.0,
            gamma=10.0,
            sticky_attack=30,
            sticky_jump=10,
            pitch_limit=(-60, 60),
            logs=False,
        )
        env = dreamerv3.wrap_env(env, config)
        return env

    from _common import run_training
    run_training(config, make_env)


if __name__ == "__main__":
    main()
