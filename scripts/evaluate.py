"""Load a trained DreamerV3 checkpoint and roll out evaluation episodes.

This module is both a CLI script and a library. Notebooks and other
callers can import :class:`Evaluator` directly::

    from scripts.evaluate import Evaluator

    evaluator = Evaluator(
        task="gym_CartPole-v1",
        preset="size1m",
        logdir="/root/logdir/cartpole",
    )
    returns = evaluator.run(episodes=5)
    print(f"mean={returns.mean():.2f}")

The CLI entry point is preserved for backwards compatibility::

    python scripts/evaluate.py \\
        --task gym_CartPole-v1 \\
        --logdir ~/logdir/cartpole \\
        --episodes 10
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def _import_make_env():
    """Import ``make_env`` whether the module is loaded as ``scripts.evaluate``
    (library use) or as a top-level script."""
    try:
        from .env_builders import make_env  # package-relative
    except ImportError:
        from env_builders import make_env  # sys.path includes scripts/
    return make_env


class Evaluator:
    """Load a DreamerV3 checkpoint and roll out evaluation episodes.

    Parameters
    ----------
    task : str
        Task spec of the form ``"<suite>_<name>"`` (e.g. ``"gym_CartPole-v1"``).
    preset : str
        DreamerV3 size preset (``"size1m"``, ``"size12m"``, ...).
    logdir : str | pathlib.Path
        Directory containing ``checkpoint.ckpt``.
    config : dreamerv3.embodied.Config, optional
        Pre-built config (used by the CLI to fold in ``embodied.Flags``
        overrides). When omitted, ``preset`` is applied on top of the
        DreamerV3 defaults.
    """

    def __init__(self, task, preset, logdir, *, config=None):
        import dreamerv3
        from dreamerv3 import embodied

        make_env = _import_make_env()

        if config is None:
            config = embodied.Config(dreamerv3.Agent.configs["defaults"])
            if preset not in dreamerv3.Agent.configs:
                raise ValueError(
                    f"Unknown preset {preset!r}. "
                    f"Known: {sorted(dreamerv3.Agent.configs)}"
                )
            config = config.update(dreamerv3.Agent.configs[preset])
            config = config.update({"logdir": str(logdir)})

        self.task = task
        self.preset = preset
        self.config = config
        self.logdir = embodied.Path(config.logdir)

        ckpt_path = self.logdir / "checkpoint.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                f"Train an agent first with scripts/train.py."
            )

        self.env = embodied.BatchEnv([make_env(task, config)], parallel=False)
        self.agent = dreamerv3.Agent(self.env.obs_space, self.env.act_space, config)

        checkpoint = embodied.Checkpoint(ckpt_path)
        checkpoint.agent = self.agent
        checkpoint.load(keys=["agent"])

    def run(self, episodes=10, *, verbose=True):
        """Run ``episodes`` eval-mode rollouts.

        Returns
        -------
        numpy.ndarray
            Shape ``(episodes,)`` of undiscounted episode returns.
        """
        import numpy as np

        returns = []
        for ep in range(episodes):
            obs = self.env.reset()
            state = None
            total = 0.0
            done = False
            while not done:
                action, state = self.agent.policy(obs, state, mode="eval")
                obs = self.env.step(action)
                total += float(obs["reward"][0])
                done = bool(obs["is_last"][0])
            returns.append(total)
            if verbose:
                print(f"episode {ep + 1}/{episodes}: return={total:.2f}")

        return np.array(returns)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--preset", default="size12m", type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--episodes", default=10, type=int)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    import dreamerv3
    from dreamerv3 import embodied

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs[args.preset])
    config = config.update({"logdir": args.logdir})
    sys.argv = [sys.argv[0]] + remaining
    config = embodied.Flags(config).parse()

    evaluator = Evaluator(args.task, args.preset, args.logdir, config=config)
    returns = evaluator.run(episodes=args.episodes)

    print(
        f"\nMean return: {returns.mean():.3f} +/- {returns.std():.3f} "
        f"(n={len(returns)})"
    )


if __name__ == "__main__":
    main()
