"""Record videos of a trained DreamerV3 agent.

This module is both a CLI script and a library. Notebooks and other
callers can import :class:`Recorder` directly::

    from scripts.record import Recorder

    rec = Recorder(
        task="gym_CartPole-v1",
        preset="size1m",
        logdir="/root/logdir/cartpole",
    )
    results = rec.record(episodes=3, fps=30)
    for r in results:
        print(r["path"], r["return"])

The CLI entry point is preserved::

    python scripts/record.py \\
        --task gym_CartPole-v1 \\
        --preset size1m \\
        --logdir ~/logdir/cartpole \\
        --episodes 3

For Gym classic-control tasks (CartPole, MountainCar, LunarLander, etc.)
the underlying env's ``render(mode='rgb_array')`` is used. For pixel-obs
tasks (DMC, Atari, Crafter, MiniGrid, Minecraft) frames come straight
from the ``image`` observation.

Requirements beyond the base install:
    pip install imageio imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def _import_make_env():
    """Import ``make_env`` whether the module is loaded as ``scripts.record``
    (library use) or as a top-level script."""
    try:
        from .env_builders import make_env  # package-relative
    except ImportError:
        from env_builders import make_env  # sys.path includes scripts/
    return make_env


class Recorder:
    """Load a DreamerV3 checkpoint and write rollout videos to disk.

    Parameters
    ----------
    task : str
        Task spec of the form ``"<suite>_<name>"``.
    preset : str
        DreamerV3 size preset.
    logdir : str | pathlib.Path
        Directory containing ``checkpoint.ckpt``.
    config : dreamerv3.embodied.Config, optional
        Pre-built config (used by the CLI to fold in user flag overrides).
    """

    def __init__(self, task, preset, logdir, *, config=None):
        import numpy as np

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

        base_env, info = make_env(
            task, config, render_mode="rgb_array", return_info=True
        )
        self.env = embodied.BatchEnv([base_env], parallel=False)

        # Pick the right frame source: pixel obs for image-based envs,
        # underlying gym.render() for vector-obs Gym envs.
        if info["obs_key"] == "image":
            def frame_fn(obs):
                return np.asarray(obs["image"][0], dtype=np.uint8)
        else:
            raw_env = info["raw_env"]
            def frame_fn(_obs):
                return np.asarray(raw_env.render(), dtype=np.uint8)
        self._frame_fn = frame_fn

        self.agent = dreamerv3.Agent(self.env.obs_space, self.env.act_space, config)

        checkpoint = embodied.Checkpoint(ckpt_path)
        checkpoint.agent = self.agent
        checkpoint.load(keys=["agent"])

    def record(self, episodes=3, *, fps=30, output_dir=None, verbose=True):
        """Roll out ``episodes`` eval-mode episodes and save one MP4 each.

        Returns
        -------
        list of dict
            One entry per episode with keys ``path`` (``pathlib.Path``),
            ``return`` (float), and ``frames`` (int).
        """
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError(
                "scripts/record.py needs imageio. Install it with:\n"
                "    pip install imageio imageio-ffmpeg"
            ) from exc

        output_dir = (
            Path(output_dir) if output_dir else Path(str(self.logdir)) / "videos"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for ep in range(episodes):
            frames = []
            obs = self.env.reset()
            frames.append(self._frame_fn(obs))
            state = None
            total_reward = 0.0
            done = False
            while not done:
                action, state = self.agent.policy(obs, state, mode="eval")
                obs = self.env.step(action)
                frames.append(self._frame_fn(obs))
                total_reward += float(obs["reward"][0])
                done = bool(obs["is_last"][0])

            video_path = output_dir / f"{self.task}_ep{ep + 1:03d}.mp4"
            imageio.mimsave(str(video_path), frames, fps=fps, macro_block_size=1)
            results.append(
                {"path": video_path, "return": total_reward, "frames": len(frames)}
            )
            if verbose:
                print(
                    f"episode {ep + 1}/{episodes}: "
                    f"return={total_reward:.2f} frames={len(frames)} -> {video_path}"
                )

        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--preset", default="size12m", type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--episodes", default=3, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Override output directory. Defaults to <logdir>/videos.",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    try:
        import imageio  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scripts/record.py needs imageio. Install it with:\n"
            "    pip install imageio imageio-ffmpeg"
        ) from exc

    import dreamerv3
    from dreamerv3 import embodied

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs[args.preset])
    config = config.update({"logdir": args.logdir})
    sys.argv = [sys.argv[0]] + remaining
    config = embodied.Flags(config).parse()

    rec = Recorder(args.task, args.preset, args.logdir, config=config)
    results = rec.record(
        episodes=args.episodes, fps=args.fps, output_dir=args.output
    )

    if results:
        mean = sum(r["return"] for r in results) / len(results)
        output_dir = (
            Path(args.output) if args.output else Path(str(rec.logdir)) / "videos"
        )
        print(f"\nMean return over {len(results)} episodes: {mean:.2f}")
        print(f"Videos saved under: {output_dir}")


if __name__ == "__main__":
    main()
