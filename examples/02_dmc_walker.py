"""Train DreamerV3 on DeepMind Control Suite — Walker Walk from pixels.

This demonstrates continuous-control learning from 64x64 image observations,
the standard DMC benchmark used in the DreamerV3 paper.

Prerequisites:
    pip install dm_control

Run (GPU strongly recommended):
    python examples/02_dmc_walker.py --logdir ~/logdir/dmc_walker

To try a different task, edit DOMAIN / TASK below. Popular choices:
    - cartpole / swingup         (easy)
    - walker   / walk            (medium, used here)
    - cheetah  / run             (medium)
    - quadruped/ run             (hard)
    - humanoid / walk            (hard)
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")

DOMAIN = "walker"
TASK = "walk"


def main():
    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import dmc

    # ------------------------------------------------------------------
    # 1. Config: defaults + 12M preset + DMC-proprio/pixel overrides.
    # ------------------------------------------------------------------
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size12m"])
    config = config.update(
        {
            "logdir": f"~/logdir/dmc_{DOMAIN}_{TASK}",
            "run.train_ratio": 512,  # DMC paper value
            "run.log_every": 120,
            "run.steps": 1_000_000,
            "batch_size": 16,
            "batch_length": 64,
            # Pixel obs only.
            "enc.simple.cnn_keys": "image",
            "dec.simple.cnn_keys": "image",
            "enc.simple.mlp_keys": "$^",
            "dec.simple.mlp_keys": "$^",
        }
    )
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    def make_env(index: int):
        env = dmc.DMC(f"{DOMAIN}_{TASK}", repeat=2, size=(64, 64))
        env = dreamerv3.wrap_env(env, config)
        return env

    env = embodied.BatchEnv([make_env(0)], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    replay = embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=logdir / "replay",
    )

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
