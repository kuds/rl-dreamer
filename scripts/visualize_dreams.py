"""Visualize DreamerV3's world model dreams.

This module is both a CLI script and a library. Notebooks and other
callers can import :class:`DreamVisualizer` directly instead of shelling
out::

    from scripts.visualize_dreams import DreamVisualizer

    viz = DreamVisualizer(
        task="crafter_reward",
        preset="size25m",
        logdir="/root/logdir/crafter",
    )
    saved = viz.generate(fps=10)

The CLI entry point is preserved for backwards compatibility::

    python scripts/visualize_dreams.py \\
        --task crafter_reward \\
        --preset size25m \\
        --logdir ~/logdir/crafter

Both paths call into the same class, so behaviour stays in sync.

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
# Low-level save helpers (shared by the class and the fallback path)
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


def save_side_by_side(real, dream, path, fps, label_left="env", label_right="dream"):
    """Write a side-by-side video of two ``[T, H, W, C]`` arrays.

    The left panel shows the real env frames from the batch; the right
    panel shows the model's open-loop reconstruction. Both arrays must
    have the same length and height; widths may differ. A 2-px black
    separator is drawn between them.
    """
    import imageio
    import numpy as np

    real = _to_uint8(np.asarray(real))
    dream = _to_uint8(np.asarray(dream))

    T = min(real.shape[0], dream.shape[0])
    real, dream = real[:T], dream[:T]

    if real.shape[1] != dream.shape[1]:
        # Pad the shorter one vertically so heights match.
        H = max(real.shape[1], dream.shape[1])
        real = _pad_to_height(real, H)
        dream = _pad_to_height(dream, H)

    sep = np.zeros((T, real.shape[1], 2, real.shape[3]), dtype=np.uint8)
    combined = np.concatenate([real, sep, dream], axis=2)
    imageio.mimsave(str(path), list(combined), fps=fps, macro_block_size=1)


def _pad_to_height(arr, H):
    import numpy as np

    pad = H - arr.shape[1]
    if pad <= 0:
        return arr
    top = pad // 2
    bot = pad - top
    return np.pad(arr, ((0, 0), (top, bot), (0, 0), (0, 0)))


def save_observation_video(batch, output_dir, fps, sheet_cols):
    """Fallback: save raw observation frames when ``report()`` has nothing.

    Looks for an ``image`` key in the batch. If the task only has vector
    observations, nothing is written.
    """
    import numpy as np

    saved = []
    if "image" not in batch:
        return saved

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
# Best-checkpoint selection
# ------------------------------------------------------------------


def _parse_step_from_name(path):
    """Pull the trailing step number out of ``checkpoint_<step>.ckpt``.

    Only matches patterns where the stem ends with ``_<digits>`` or is
    purely digits, so the rolling ``checkpoint.ckpt`` (no number) and
    rotated names like ``checkpoint.1.ckpt`` (small index, not a step)
    are not mistaken for step snapshots.
    """
    import re

    m = re.match(r"(?:.*_)?(\d{3,})$", path.stem)
    return int(m.group(1)) if m else None


def _list_checkpoints(logdir):
    """Return all ``*.ckpt`` files in ``logdir``."""
    return sorted(Path(logdir).glob("*.ckpt"))


def _best_step_from_metrics(logdir, return_keys=("episode/score", "eval/return", "train/return")):
    """Scan ``metrics.jsonl`` for the step with the highest episode return.

    Returns ``(step, value, key)`` or ``None`` if metrics are missing /
    contain none of the expected keys.
    """
    import json

    path = Path(logdir) / "metrics.jsonl"
    if not path.exists():
        return None

    best = None
    chosen_key = None
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step") or record.get("global_step")
                if step is None:
                    continue
                for key in return_keys:
                    if key in record:
                        value = float(record[key])
                        if best is None or value > best[1]:
                            best = (int(step), value, key)
                            chosen_key = key
                        break
    except OSError:
        return None
    return best


def select_best_checkpoint(logdir):
    """Pick the highest-performing checkpoint under ``logdir``.

    Strategy, in order:

    1. If snapshot checkpoints (``checkpoint_<step>.ckpt``) exist and
       ``metrics.jsonl`` has episode-return entries, return the snapshot
       whose step is closest to the best-return step.
    2. If snapshots exist but no metrics, return the latest snapshot by
       step number (most-trained ≈ best, on average).
    3. Otherwise, fall back to ``checkpoint.ckpt``.

    Returns
    -------
    (pathlib.Path | None, str)
        ``(path, source)`` where ``source`` is one of
        ``"best-return"``, ``"latest-step"``, or ``"checkpoint.ckpt"``.
        ``path`` is ``None`` if no checkpoint exists.
    """
    logdir = Path(logdir)
    snapshots = [
        p for p in _list_checkpoints(logdir)
        if _parse_step_from_name(p) is not None
    ]

    if snapshots:
        best = _best_step_from_metrics(logdir)
        if best is not None:
            best_step = best[0]
            chosen = min(snapshots, key=lambda p: abs(_parse_step_from_name(p) - best_step))
            return chosen, f"best-return ({best[2]}={best[1]:.3f} @ step {best_step})"
        # No metrics: take the most-trained snapshot.
        latest = max(snapshots, key=_parse_step_from_name)
        return latest, "latest-step"

    rolling = logdir / "checkpoint.ckpt"
    if rolling.exists():
        return rolling, "checkpoint.ckpt"
    return None, "missing"


# ------------------------------------------------------------------
# Library entry point
# ------------------------------------------------------------------


class DreamVisualizer:
    """Load a DreamerV3 checkpoint and render world-model dreams.

    The class wraps the full pipeline:

    1. Build the env and restore the agent from ``<logdir>/checkpoint.ckpt``.
    2. Sample a batch from the replay buffer under ``<logdir>/replay/``,
       or — if none exists — collect a live rollout.
    3. Call ``agent.report(batch)`` and persist every visual output to
       ``<logdir>/dreams/`` (or a caller-supplied directory).

    Parameters
    ----------
    task : str
        Task spec of the form ``"<suite>_<name>"`` (e.g. ``"crafter_reward"``).
    preset : str
        DreamerV3 size preset (``"size1m"``, ``"size12m"``, ...).
    logdir : str | pathlib.Path
        Directory containing the checkpoint(s) and (optionally) a
        ``replay/`` subdirectory.
    config : dreamerv3.embodied.Config, optional
        Pre-built config. Use this when calling from a CLI entry point
        that already folded in user ``embodied.Flags`` overrides. When
        omitted, ``preset`` is applied on top of the DreamerV3 defaults.
    checkpoint : str | pathlib.Path, optional
        Explicit checkpoint file to load. When omitted, the
        best-performing snapshot is auto-selected (see
        :func:`select_best_checkpoint`).
    """

    def __init__(self, task, preset, logdir, *, config=None, checkpoint=None):
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

        if checkpoint is None:
            ckpt_path, source = select_best_checkpoint(Path(str(self.logdir)))
        else:
            ckpt_path = Path(checkpoint)
            source = "explicit"
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found under {self.logdir}. "
                f"Train an agent first with scripts/train.py."
            )
        self.checkpoint_path = ckpt_path
        self.checkpoint_source = source
        print(f"loading checkpoint from {ckpt_path} (selected by: {source})")

        self.env = embodied.BatchEnv([make_env(task, config)], parallel=False)
        self.agent = dreamerv3.Agent(self.env.obs_space, self.env.act_space, config)

        ckpt = embodied.Checkpoint(embodied.Path(str(ckpt_path)))
        ckpt.agent = self.agent
        ckpt.load(keys=["agent"])
        self.batch_source = None

    def load_batch_from_replay(self):
        """Try to sample a batch from the saved replay buffer.

        Returns a dict of numpy arrays with shape ``[B, T, ...]`` or
        ``None`` if the replay directory is missing / unreadable.
        """
        from dreamerv3 import embodied

        replay_dir = self.logdir / "replay"
        if not replay_dir.exists():
            return None
        try:
            replay = embodied.replay.Replay(
                length=self.config.batch_length,
                capacity=self.config.replay.size,
                directory=replay_dir,
            )
            dataset = replay.dataset(batch=self.config.batch_size)
            return next(iter(dataset))
        except Exception as exc:
            print(f"[warn] could not load batch from replay: {exc}")
            return None

    def collect_batch(self, length=None):
        """Run a live rollout and format it as a ``[1, T, ...]`` batch dict."""
        import numpy as np

        if length is None:
            length = self.config.batch_length

        obs = self.env.reset()
        state = None
        steps = []

        first = {k: np.asarray(v[0]) for k, v in obs.items()}
        act_space = self.env.act_space["action"]
        first["action"] = np.zeros(act_space.shape, dtype=act_space.dtype)
        steps.append(first)

        for t in range(1, length):
            action, state = self.agent.policy(obs, state, mode="eval")
            obs = self.env.step(action)
            step = {k: np.asarray(v[0]) for k, v in obs.items()}
            step["action"] = np.asarray(action["action"][0])
            steps.append(step)
            if bool(obs["is_last"][0]) and t < length - 1:
                obs = self.env.reset()
                state = None

        return {k: np.stack([s[k] for s in steps], axis=0)[np.newaxis] for k in steps[0]}

    def get_batch(self):
        """Return a batch, preferring replay and falling back to a rollout."""
        batch = self.load_batch_from_replay()
        if batch is None:
            batch = self.collect_batch()
            self.batch_source = "rollout"
        else:
            self.batch_source = "replay"
        return batch

    def generate(self, output_dir=None, *, fps=10, sheet_cols=10, batch=None):
        """Run ``agent.report()`` on a batch and save videos + sheets.

        Parameters
        ----------
        output_dir : str | pathlib.Path, optional
            Where to write MP4s and PNGs. Defaults to
            ``<logdir>/dreams``.
        fps : int
            Frames per second for the output MP4s.
        sheet_cols : int
            Number of columns in the contact-sheet PNGs.
        batch : dict, optional
            Pre-built batch to use. When omitted, one is drawn from
            replay / a live rollout via :meth:`get_batch`.

        Returns
        -------
        list of (str, pathlib.Path, tuple)
            One entry per file written: ``(key, path, shape)``.
        """
        if batch is None:
            batch = self.get_batch()

        output_dir = Path(output_dir) if output_dir else Path(str(self.logdir)) / "dreams"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        try:
            report = self.agent.report(batch)
            saved = save_report_outputs(
                report, output_dir, fps=fps, sheet_cols=sheet_cols
            )
        except Exception as exc:
            print(f"[warn] agent.report() failed: {exc}")

        if not saved:
            saved = save_observation_video(
                batch, output_dir, fps=fps, sheet_cols=sheet_cols
            )

        return saved

    def generate_side_by_side(
        self, output_dir=None, *, fps=10, batch=None, dream_key=None
    ):
        """Render real env frames next to the model's dream frames.

        Pulls the ``image`` stream out of the batch fed to
        ``agent.report()`` and pairs it horizontally with the model's
        open-loop reconstruction. Returns the output path, or ``None``
        if neither stream is available (e.g. vector-only tasks, or a
        report without an open-loop video).

        Parameters
        ----------
        output_dir : str | pathlib.Path, optional
            Where to write the MP4. Defaults to ``<logdir>/dreams``.
        fps : int
            Frames per second.
        batch : dict, optional
            Pre-built batch. When omitted, one is drawn via
            :meth:`get_batch`.
        dream_key : str, optional
            Specific ``agent.report()`` key to use as the dream stream.
            When ``None``, the first key matching ``openl*`` is used,
            falling back to ``recon*``.
        """
        import numpy as np

        if batch is None:
            batch = self.get_batch()

        if "image" not in batch:
            print(
                "[warn] batch has no 'image' stream — side-by-side visualization "
                "needs a pixel-based task (DMC, Atari, Crafter, MiniGrid, Minecraft)."
            )
            return None

        try:
            report = self.agent.report(batch)
        except Exception as exc:
            print(f"[warn] agent.report() failed: {exc}")
            return None

        dream = _select_dream_stream(report, dream_key)
        if dream is None:
            print(
                "[warn] no openl/recon video stream found in report; "
                f"available keys: {sorted(report)}"
            )
            return None

        real = np.asarray(batch["image"])
        if real.ndim == 5:
            real = real[0]

        output_dir = (
            Path(output_dir) if output_dir else Path(str(self.logdir)) / "dreams"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "side_by_side.mp4"

        save_side_by_side(real, dream, out_path, fps=fps)
        return out_path


def _select_dream_stream(report, dream_key=None):
    """Pull a ``[T, H, W, C]`` frame stack out of ``agent.report()``."""
    import numpy as np

    candidates = [dream_key] if dream_key else None
    if candidates is None:
        openl = sorted(k for k in report if "openl" in k.lower())
        recon = sorted(k for k in report if "recon" in k.lower())
        candidates = openl + recon

    for key in candidates:
        if key not in report:
            continue
        arr = np.asarray(report[key])
        if arr.ndim == 5 and arr.shape[-1] in (1, 3, 4):
            arr = arr[0]
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            return arr
    return None


def _import_make_env():
    """Import ``make_env`` whether this module is loaded as ``scripts.visualize_dreams``
    (notebook / library use) or as a top-level script."""
    try:
        from .env_builders import make_env  # package-relative
    except ImportError:
        from env_builders import make_env  # sys.path includes scripts/
    return make_env


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
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Also write side_by_side.mp4 pairing real env frames with the dream.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help=(
            "Explicit checkpoint file. When omitted, the highest-return "
            "snapshot is auto-selected from <logdir>."
        ),
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    try:
        import imageio  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scripts/visualize_dreams.py needs imageio. Install it with:\n"
            "    pip install imageio imageio-ffmpeg"
        ) from exc

    import dreamerv3
    from dreamerv3 import embodied

    # Build the config so user-supplied --foo.bar flags still flow through.
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

    viz = DreamVisualizer(
        args.task,
        args.preset,
        args.logdir,
        config=config,
        checkpoint=args.checkpoint,
    )

    saved = viz.generate(
        output_dir=args.output,
        fps=args.fps,
        sheet_cols=args.sheet_cols,
    )

    if args.side_by_side:
        sbs_path = viz.generate_side_by_side(output_dir=args.output, fps=args.fps)
        if sbs_path is not None:
            saved.append(("side_by_side", sbs_path, None))

    if not saved:
        print(
            "no image observations found in batch. "
            "Dream visualization requires a pixel-based task "
            "(DMC, Atari, Crafter, MiniGrid, Minecraft)."
        )
        return

    output_dir = Path(args.output) if args.output else Path(str(viz.logdir)) / "dreams"
    print(f"\n{'='*60}")
    print(f"  {len(saved)} file(s) written to {output_dir}")
    print(f"  data source: {viz.batch_source}")
    print(f"{'='*60}")
    for key, path, shape in saved:
        suffix = Path(path).suffix
        kind = "video" if suffix == ".mp4" else "image"
        print(f"  [{kind:5s}] {key:30s} {str(shape):20s} → {Path(path).name}")
    print()


if __name__ == "__main__":
    main()
