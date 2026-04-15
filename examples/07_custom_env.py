"""Train DreamerV3 on a custom Gymnasium environment.

Use this file as a template when you want to plug in your own environment.
The minimal requirements for DreamerV3 are:

    - Observation space: `gym.spaces.Box` (vector or image) or a `Dict`
      of Box spaces. Images must be `uint8` HxWxC.
    - Action space: `Box` (continuous) or `Discrete`.
    - Standard Gymnasium 5-tuple `step` API:
          obs, reward, terminated, truncated, info

Run:
    python examples/07_custom_env.py --logdir ~/logdir/custom
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


# ----------------------------------------------------------------------
# A tiny example environment: navigate a 1D track to reach a goal.
# Replace this class with your own env.
# ----------------------------------------------------------------------
class OneDNavigation:
    """Minimal Gymnasium-compatible 1D navigation task."""

    def __init__(self, length: int = 10, max_steps: int = 50):
        import gymnasium as gym

        self._length = length
        self._max_steps = max_steps
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # left, stay, right
        self._pos = 0
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._pos = 0
        self._t = 0
        return self._obs(), {}

    def step(self, action: int):
        delta = {0: -1, 1: 0, 2: 1}[int(action)]
        self._pos = int(np.clip(self._pos + delta, 0, self._length))
        self._t += 1

        reached = self._pos == self._length
        reward = 1.0 if reached else -0.01
        terminated = reached
        truncated = self._t >= self._max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return np.array(
            [self._pos / self._length, self._t / self._max_steps],
            dtype=np.float32,
        )


def main():
    import dreamerv3
    from dreamerv3 import embodied
    from dreamerv3.embodied.envs import from_gym

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs["size1m"])
    config = config.update(
        {
            "logdir": "~/logdir/custom",
            "run.train_ratio": 32,
            "run.log_every": 30,
            "run.steps": 20_000,
            "batch_size": 16,
            "batch_length": 32,
            "enc.simple.mlp_keys": "vector",
            "dec.simple.mlp_keys": "vector",
            "enc.simple.cnn_keys": "$^",
            "dec.simple.cnn_keys": "$^",
            "jax.platform": "cpu",
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
        env = OneDNavigation()
        env = from_gym.FromGym(env, obs_key="vector")
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
