"""Visualize DreamerV3's world model dreams.

Loads a trained checkpoint and produces videos showing what the world
model "sees" (posterior reconstructions) and "dreams" (open-loop
imagination from the prior).  This complements ``scripts/record.py``
which only shows the agent's behaviour in the *real* environment.

The script calls ``agent.report(batch)`` — DreamerV3's standard hook
for producing eval-time visualizations.  It first tries to load a
batch from the saved replay buffer under ``<logdir>/replay/``; if none
exists, it falls back to collecting a short rollout from the live
environment.

Every video-like array returned by ``report()`` is saved as an MP4 and
as a contact-sheet PNG (a grid of sampled frames for quick inspection).

Example:
    python scripts/visualize_dreams.py \\
        --task crafter_reward \\
        --preset size25m \\
        --logdir ~/logdir/crafter

    python scripts/visualize_dreams.py \\
        --task dmc_walker_walk \\
        --preset size12m \\
        --logdir ~/logdir/walker \\
        --fps 15

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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DreamerV3 world-model reconstructions and dreams.",
        add_help=False,
    )
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--preset", default="size12m", type=str)
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument(
        "--fps",
        default=10,
        type=int,
        help="Frames per second for output MP4s (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Override output directory. Defaults to <logdir>/dreams.",
    )
    parser.add_argument(
        "--sheet-cols",
        default=10,
        type=int,
        help="Columns in the contact-sheet PNG (default: 10).",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------


def load_batch_from_replay(logdir, config):
    """Try to sample a single batch from the saved replay buffer.

    Returns a dict of numpy arrays with shape ``[B, T, ...]`` or
    ``None`` if the replay directory is missing / empty.
    """
    from dreamerv3 import embodied

    replay_dir = logdir / "replay"
    if not replay_dir.exists():
        return None
    try:
        replay = embodied.replay.Replay(
            length=config.batch_length,
            capacity=config.replay.size,
            directory=replay_dir,
        )
        # The dataset yields dicts of arrays with shape [B, T, ...].
        dataset = replay.dataset(batch=config.batch_size)
        batch = next(iter(dataset))
        return batch
    except Exception as exc:
        print(f"[warn] could not load batch from replay: {exc}")
        return None


def collect_batch(env, agent, length):
    """Run a rollout and format it as a ``[1, T, ...]`` batch dict.

    This is the fallback used when no replay data is available.
    """
    import numpy as np

    obs = env.reset()
    state = None

    steps = []

    # --- first timestep (is_first=True, action=zeros) ---
    first = {}
    for k, v in obs.items():
        first[k] = np.asarray(v[0])  # strip env batch dim
    act_space = env.act_space["action"]
    first["action"] = np.zeros(act_space.shape, dtype=act_space.dtype)
    steps.append(first)

    for t in range(1, length):
        action, state = agent.policy(obs, state, mode="eval")
        obs = env.step(action)

        step = {}
        for k, v in obs.items():
            step[k] = np.asarray(v[0])
        step["action"] = np.asarray(action["action"][0])
        steps.append(step)

        if bool(obs["is_last"][0]):
            if t < length - 1:
                obs = env.reset()
                state = None

    # Stack to [1, T, ...]
    batch = {}
    for k in steps[0]:
        batch[k] = np.stack([s[k] for s in steps], axis=0)[np.newaxis]
    return batch


# ------------------------------------------------------------------
# Video / image saving
# ------------------------------------------------------------------


def _to_uint8(arr):
    """Convert an array to uint8, handling both [0,1] and [0,255] ranges."""
    import numpy as np

    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return arr.clip(0, 255).astype(np.uint8)


def save_video(frames, path, fps):
    """Write a ``[T, H, W, C]`` uint8 array as an MP4."""
    import imageio

    imageio.mimsave(str(path), list(frames), fps=fps, macro_block_size=1)


def save_contact_sheet(frames, path, cols=10):
    """Write a grid of sampled frames as a single PNG."""
    import imageio
    import numpy as np

    T, H, W, C = frames.shape
    rows = (T + cols - 1) // cols
    grid = np.zeros((rows * H, cols * W, C), dtype=frames.dtype)
    for i in range(T):
        r, c = divmod(i, cols)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = frames[i]
    imageio.imsave(str(path), grid)


def save_report_outputs(report, output_dir, fps, sheet_cols):
    """Iterate over ``agent.report()`` results and save visual outputs.

    Returns a list of ``(key, path, shape)`` tuples for everything that
    was written.
    """
    import numpy as np

    saved = []
    for key in sorted(report):
        val = report[key]
        arr = np.asarray(val)

        # ---- video: [T, H, W, C] or [B, T, H, W, C] ----
        if arr.ndim == 5 and arr.shape[-1] in (1, 3, 4):
            arr = arr[0]  # first batch element
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            arr = _to_uint8(arr)
            vid_path = output_dir / f"{key}.mp4"
            save_video(arr, vid_path, fps)
            saved.append((key, vid_path, arr.shape))
            sheet_path = output_dir / f"{key}_sheet.png"
            save_contact_sheet(arr, sheet_path, cols=sheet_cols)
            saved.append((key, sheet_path, arr.shape))
            continue

        # ---- single image: [H, W, C] ----
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            import imageio

            arr = _to_uint8(arr)
            img_path = output_dir / f"{key}.png"
            imageio.imsave(str(img_path), arr)
            saved.append((key, img_path, arr.shape))
            continue

    return saved


def save_observation_video(batch, output_dir, fps, sheet_cols):
    """Fallback: save raw observation frames when report() is unavailable.

    Looks for an ``image`` key in the batch.  If the task only has
    vector observations, nothing is written.
    """
    import numpy as np

    saved = []
    if "image" not in batch:
        return saved

    # batch["image"] has shape [B, T, H, W, C]
    frames = np.asarray(batch["image"])
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        return saved

    frames = _to_uint8(frames)
    vid_path = output_dir / "observation.mp4"
    save_video(frames, vid_path, fps)
    saved.append(("observation", vid_path, frames.shape))

    sheet_path = output_dir / "observation_sheet.png"
    save_contact_sheet(frames, sheet_path, cols=sheet_cols)
    saved.append(("observation", sheet_path, frames.shape))
    return saved


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    args, remaining = parse_args()

    try:
        import imageio  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scripts/visualize_dreams.py needs imageio. Install it with:\n"
            "    pip install imageio imageio-ffmpeg"
        ) from exc

    import numpy as np

    import dreamerv3
    from dreamerv3 import embodied

    # ---- config ----
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    if args.preset not in dreamerv3.Agent.configs:
        raise SystemExit(
            f"Unknown preset {args.preset!r}. "
            f"Known: {sorted(dreamerv3.Agent.configs)}"
        )
    config = config.update(dreamerv3.Agent.configs[args.preset])
    config = config.update({"logdir": args.logdir})
    sys.argv = [sys.argv[0]] + remaining
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)

    # ---- checkpoint ----
    ckpt_path = logdir / "checkpoint.ckpt"
    if not ckpt_path.exists():
        raise SystemExit(
            f"No checkpoint found at {ckpt_path}.\n"
            f"Train an agent first with scripts/train.py."
        )

    # We need obs_space and act_space.  Import the env builder so we can
    # create a throwaway env to query spaces, even if we end up loading
    # data from replay.
    sys.path.insert(0, str(embodied.Path(__file__).parent))
    from train import make_env  # type: ignore

    env = embodied.BatchEnv([make_env(args.task, config)], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)

    checkpoint = embodied.Checkpoint(ckpt_path)
    checkpoint.agent = agent
    checkpoint.load(keys=["agent"])
    print(f"loaded checkpoint from {ckpt_path}")

    # ---- get a batch ----
    print("preparing batch …")
    batch = load_batch_from_replay(logdir, config)
    source = "replay"
    if batch is None:
        print("no replay data found — collecting a fresh rollout")
        batch = collect_batch(env, agent, config.batch_length)
        source = "rollout"
    else:
        print(f"loaded batch from replay (keys: {sorted(batch.keys())})")

    # Show batch shapes for debugging.
    for k in sorted(batch):
        v = np.asarray(batch[k])
        print(f"  batch[{k!r}]: {v.shape} {v.dtype}")

    # ---- output directory ----
    output_dir = Path(args.output) if args.output else Path(str(logdir)) / "dreams"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- call agent.report() ----
    saved = []
    try:
        print("calling agent.report() …")
        report = agent.report(batch)
        print(f"report returned {len(report)} keys: {sorted(report.keys())}")
        saved = save_report_outputs(
            report, output_dir, fps=args.fps, sheet_cols=args.sheet_cols
        )
    except Exception as exc:
        print(f"[warn] agent.report() failed: {exc}")
        print("falling back to raw observation video")

    # ---- fallback: save raw observation frames ----
    if not saved:
        print(
            "no visual outputs from report() — "
            "saving raw observation frames instead"
        )
        saved = save_observation_video(
            batch, output_dir, fps=args.fps, sheet_cols=args.sheet_cols
        )

    if not saved:
        print(
            "no image observations found in batch. "
            "Dream visualization requires a pixel-based task "
            "(DMC, Atari, Crafter, MiniGrid, Minecraft)."
        )
        return

    # ---- summary ----
    print(f"\n{'='*60}")
    print(f"  {len(saved)} file(s) written to {output_dir}")
    print(f"  data source: {source}")
    print(f"{'='*60}")
    for key, path, shape in saved:
        suffix = path.suffix
        kind = "video" if suffix == ".mp4" else "image"
        print(f"  [{kind:5s}] {key:30s} {str(shape):20s} → {path.name}")
    print()


if __name__ == "__main__":
    main()
