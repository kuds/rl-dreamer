"""Record videos of a trained DreamerV3 agent.

Loads a checkpoint from ``--logdir`` and writes one MP4 per episode to
``<logdir>/videos/``. Supports every task suite the generic trainer knows
about.

Example:
    python scripts/record.py \
        --task gym_CartPole-v1 \
        --preset size1m \
        --logdir ~/logdir/cartpole \
        --episodes 3

    python scripts/record.py \
        --task crafter_reward \
        --preset size25m \
        --logdir ~/logdir/crafter \
        --episodes 5 \
        --fps 15

Requirements beyond the base install:
    pip install imageio imageio-ffmpeg

For Gym classic-control tasks (CartPole, MountainCar, LunarLander, etc.)
the underlying env's ``render(mode='rgb_array')`` is used. For pixel-obs
tasks (DMC, Atari, Crafter, MiniGrid, Minecraft) frames come straight
from the ``image`` observation — no extra renderer needed.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


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


def make_env_and_renderer(task: str, config):
    """Return (dreamer_env, frame_fn).

    ``frame_fn()`` yields an HxWx3 uint8 numpy array for the current env
    state. For pixel envs we read ``obs["image"]``; for vector Gym envs
    we call the underlying env's ``render()``.
    """
    import numpy as np

    import dreamerv3
    from dreamerv3 import embodied

    suite, _, name = task.partition("_")
    raw_env = None  # kept alive for vector envs that need .render()

    if suite == "gym":
        import gymnasium as gym
        from dreamerv3.embodied.envs import from_gym

        raw_env = gym.make(name, render_mode="rgb_array")
        shape = getattr(raw_env.observation_space, "shape", None)
        is_image = shape is not None and len(shape) == 3 and shape[-1] in (1, 3)
        env = from_gym.FromGym(raw_env, obs_key="image" if is_image else "vector")

        if is_image:
            def frame_fn(last_obs):
                img = last_obs["image"][0]
                return np.asarray(img, dtype=np.uint8)
        else:
            def frame_fn(last_obs):
                frame = raw_env.render()
                return np.asarray(frame, dtype=np.uint8)

    elif suite == "dmc":
        from dreamerv3.embodied.envs import dmc

        env = dmc.DMC(name, repeat=2, size=(64, 64))

        def frame_fn(last_obs):
            return np.asarray(last_obs["image"][0], dtype=np.uint8)

    elif suite == "atari":
        from dreamerv3.embodied.envs import atari

        env = atari.Atari(
            name,
            size=(64, 64),
            gray=False,
            noops=30,
            lives="unused",
            sticky=True,
            actions="all",
            length=108_000,
            resize="pillow",
        )

        def frame_fn(last_obs):
            return np.asarray(last_obs["image"][0], dtype=np.uint8)

    elif suite == "crafter":
        import crafter
        from dreamerv3.embodied.envs import from_gym

        env = crafter.Env()
        env = from_gym.FromGym(env, obs_key="image")

        def frame_fn(last_obs):
            return np.asarray(last_obs["image"][0], dtype=np.uint8)

    elif suite == "minigrid":
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
        from dreamerv3.embodied.envs import from_gym

        mg_env = gym.make(name, render_mode="rgb_array")
        mg_env = RGBImgPartialObsWrapper(mg_env, tile_size=8)
        mg_env = ImgObsWrapper(mg_env)
        env = from_gym.FromGym(mg_env, obs_key="image")

        def frame_fn(last_obs):
            return np.asarray(last_obs["image"][0], dtype=np.uint8)

    elif suite == "minecraft":
        from dreamerv3.embodied.envs import minecraft

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

        def frame_fn(last_obs):
            return np.asarray(last_obs["image"][0], dtype=np.uint8)

    else:
        raise ValueError(f"Unknown task suite: {suite!r} (from task={task!r})")

    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    return env, frame_fn


def main():
    args, remaining = parse_args()

    try:
        import imageio
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

    logdir = embodied.Path(config.logdir)
    env, frame_fn = make_env_and_renderer(args.task, config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    checkpoint.load(keys=["agent"])

    output_dir = Path(args.output) if args.output else Path(str(logdir)) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_returns = []
    for ep in range(args.episodes):
        frames = []
        obs = env.reset()
        frames.append(frame_fn(obs))
        state = None
        total_reward = 0.0
        done = False
        while not done:
            action, state = agent.policy(obs, state, mode="eval")
            obs = env.step(action)
            frames.append(frame_fn(obs))
            total_reward += float(obs["reward"][0])
            done = bool(obs["is_last"][0])

        video_path = output_dir / f"{args.task}_ep{ep + 1:03d}.mp4"
        imageio.mimsave(str(video_path), frames, fps=args.fps, macro_block_size=1)
        total_returns.append(total_reward)
        print(
            f"episode {ep + 1}/{args.episodes}: "
            f"return={total_reward:.2f} frames={len(frames)} -> {video_path}"
        )

    if total_returns:
        mean = sum(total_returns) / len(total_returns)
        print(f"\nMean return over {len(total_returns)} episodes: {mean:.2f}")
        print(f"Videos saved under: {output_dir}")


if __name__ == "__main__":
    main()
