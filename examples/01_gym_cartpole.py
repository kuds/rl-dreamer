"""Train DreamerV3 on Gymnasium's CartPole-v1.

This is the smallest possible working example — a classic-control task with
low-dimensional vector observations and a discrete action space. It runs on
CPU in a few minutes and is a good first smoke test of your installation.

Run:
    python examples/01_gym_cartpole.py --logdir ~/logdir/cartpole

DreamerV3 is massive overkill for CartPole — this is intentionally a toy
example to verify the end-to-end training loop.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def main():
    import gymnasium as gym

    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import from_gym

    # ------------------------------------------------------------------
    # 1. Build config: defaults + small preset + CartPole overrides.
    # ------------------------------------------------------------------
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size1m"])  # tiny model
    config = config.update(
        {
            "logdir": "~/logdir/cartpole",
            "run.train_ratio": 32,
            "run.log_every": 30,  # seconds
            "run.steps": 50_000,
            "batch_size": 16,
            "batch_length": 32,
            # CartPole has vector observations and no image stream.
            "enc.simple.mlp_keys": "vector",
            "dec.simple.mlp_keys": "vector",
            "enc.simple.cnn_keys": "$^",  # disable CNN encoder
            "dec.simple.cnn_keys": "$^",
            "jax.platform": "cpu",
        }
    )
    config = embodied.Flags(config).parse()

    # ------------------------------------------------------------------
    # 2. Logger, replay buffer, and env stack.
    # ------------------------------------------------------------------
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
        env = gym.make("CartPole-v1")
        env = from_gym.FromGym(env, obs_key="vector")
        env = dreamerv3.wrap_env(env, config)
        return env

    env = embodied.BatchEnv([make_env(0)], parallel=False)

    # ------------------------------------------------------------------
    # 3. Agent + replay + training loop.
    # ------------------------------------------------------------------
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
