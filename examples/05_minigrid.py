"""Train DreamerV3 on MiniGrid.

MiniGrid is a fast grid-world benchmark — a useful sanity check for
pixel-based discrete-action learning when you don't have Atari or DMC
dependencies available.

Prerequisites:
    pip install minigrid

Run:
    python examples/05_minigrid.py --logdir ~/logdir/minigrid

Swap in any MiniGrid task by editing ENV_ID. A few examples:
    - MiniGrid-Empty-5x5-v0            (easy)
    - MiniGrid-DoorKey-6x6-v0          (medium)
    - MiniGrid-MultiRoom-N4-S5-v0      (hard)
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")

ENV_ID = "MiniGrid-DoorKey-6x6-v0"


def main():
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers envs)
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import from_gym

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size12m"])
    config = config.update(
        {
            "logdir": f"~/logdir/minigrid_{ENV_ID}",
            "run.train_ratio": 32,
            "run.log_every": 60,
            "run.steps": 500_000,
            "batch_size": 16,
            "batch_length": 64,
            "enc.simple.cnn_keys": "image",
            "dec.simple.cnn_keys": "image",
            "enc.simple.mlp_keys": "$^",
            "dec.simple.mlp_keys": "$^",
        }
    )
    config = embodied.Flags(config).parse()

    def make_env(index: int):
        env = gym.make(ENV_ID)
        # Use RGB pixel observations and drop the language mission string.
        env = RGBImgPartialObsWrapper(env, tile_size=8)
        env = ImgObsWrapper(env)
        env = from_gym.FromGym(env, obs_key="image")
        env = dreamerv3.wrap_env(env, config)
        return env

    from _common import run_training
    run_training(config, make_env)


if __name__ == "__main__":
    main()
