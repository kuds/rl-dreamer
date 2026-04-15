"""Train DreamerV3 on Crafter — a 2D open-world survival benchmark.

Crafter is the benchmark on which DreamerV3 was the first algorithm to
collect diamonds from scratch. The environment has sparse rewards,
long-horizon structure, and 22 achievements.

Prerequisites:
    pip install crafter

Run (GPU recommended, but Crafter also works on CPU):
    python examples/04_crafter.py --logdir ~/logdir/crafter

The paper uses `size200m` for 1e8 env steps. The defaults below use `size25m`
so the run fits on a single commodity GPU in reasonable time.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def main():
    import crafter

    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import from_gym

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size25m"])
    config = config.update(
        {
            "logdir": "~/logdir/crafter",
            "run.train_ratio": 64,
            "run.log_every": 120,
            "run.steps": 1_000_000,
            "batch_size": 16,
            "batch_length": 64,
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
        env = crafter.Env()  # Crafter reward variant by default
        env = from_gym.FromGym(env, obs_key="image")
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
