"""Train DreamerV3 on Atari Pong via the Arcade Learning Environment.

Prerequisites:
    pip install "gymnasium[atari,accept-rom-license]" ale-py

Run (GPU strongly recommended):
    python examples/03_atari_pong.py --logdir ~/logdir/atari_pong

Swap in any other Atari game by editing GAME (e.g. 'breakout', 'boxing',
'ms_pacman'). For the Atari 100k benchmark, use `size25m` and
`run.steps = 400_000` (env steps, with action repeat 4 -> 100k agent steps).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")

GAME = "pong"


def main():
    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import atari

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size25m"])
    config = config.update(
        {
            "logdir": f"~/logdir/atari_{GAME}",
            "run.train_ratio": 64,
            "run.log_every": 120,
            "run.steps": 400_000,  # Atari 100k (with action repeat 4)
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
        env = atari.Atari(
            GAME,
            size=(64, 64),
            gray=False,
            noops=30,
            lives="unused",
            sticky=True,
            actions="all",
            length=108_000,
            resize="pillow",
        )
        env = dreamerv3.wrap_env(env, config)
        return env

    from _common import run_training
    run_training(config, make_env)


if __name__ == "__main__":
    main()
