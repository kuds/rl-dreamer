"""Generic DreamerV3 training entry point.

This module is both a CLI script and a library. Notebooks and other
callers can import :class:`Trainer` directly::

    from scripts.train import Trainer

    trainer = Trainer(
        task="gym_CartPole-v1",
        preset="size1m",
        logdir="/root/logdir/cartpole",
        overrides={"run.steps": 20_000},
    )
    trainer.run()

The CLI entry point is preserved::

    python scripts/train.py --task gym_CartPole-v1 --preset size1m
    python scripts/train.py --task dmc_walker_walk --preset size12m
    python scripts/train.py --task atari_pong --preset size25m
    python scripts/train.py --task crafter_reward --preset size25m
    python scripts/train.py --task minigrid_MiniGrid-DoorKey-6x6-v0
    python scripts/train.py --task minecraft_diamond --preset size25m

Both paths build the same config and call the same training loop.
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


# Per-suite config defaults. These mirror what the standalone
# ``examples/0X_*.py`` scripts hard-code, kept in one place so notebook
# and CLI users don't have to remember which encoder/decoder keys each
# suite needs.
_SUITE_DEFAULTS = {
    "gym": {
        "run.train_ratio": 32,
    },
    "dmc": {
        "run.train_ratio": 512,
        "enc.simple.cnn_keys": "image",
        "dec.simple.cnn_keys": "image",
        "enc.simple.mlp_keys": "$^",
        "dec.simple.mlp_keys": "$^",
    },
    "atari": {
        "run.train_ratio": 64,
        "enc.simple.cnn_keys": "image",
        "dec.simple.cnn_keys": "image",
        "enc.simple.mlp_keys": "$^",
        "dec.simple.mlp_keys": "$^",
    },
    "crafter": {
        "run.train_ratio": 64,
        "enc.simple.cnn_keys": "image",
        "dec.simple.cnn_keys": "image",
        "enc.simple.mlp_keys": "$^",
        "dec.simple.mlp_keys": "$^",
    },
    "minigrid": {
        "run.train_ratio": 32,
        "enc.simple.cnn_keys": "image",
        "dec.simple.cnn_keys": "image",
        "enc.simple.mlp_keys": "$^",
        "dec.simple.mlp_keys": "$^",
    },
    "minecraft": {
        "run.train_ratio": 16,
        "enc.simple.cnn_keys": "image",
        "dec.simple.cnn_keys": "image",
        "enc.simple.mlp_keys": "$^",
        "dec.simple.mlp_keys": "$^",
    },
}

# Vector-obs gym tasks need MLP-only encoders/decoders. ``Trainer``
# detects pixel vs vector obs automatically for the ``gym`` suite.
_GYM_VECTOR_OVERRIDES = {
    "enc.simple.mlp_keys": "vector",
    "dec.simple.mlp_keys": "vector",
    "enc.simple.cnn_keys": "$^",
    "dec.simple.cnn_keys": "$^",
}
_GYM_PIXEL_OVERRIDES = {
    "enc.simple.cnn_keys": "image",
    "dec.simple.cnn_keys": "image",
    "enc.simple.mlp_keys": "$^",
    "dec.simple.mlp_keys": "$^",
}


def _import_make_env():
    """Import ``make_env`` whether the module is loaded as ``scripts.train``
    (library use) or as a top-level script."""
    try:
        from .env_builders import make_env  # package-relative
    except ImportError:
        from env_builders import make_env  # sys.path includes scripts/
    return make_env


def _detect_gym_obs_kind(task):
    """Return ``"image"`` or ``"vector"`` for a ``gym_<name>`` task.

    Falls back to ``"vector"`` if Gymnasium isn't importable so we don't
    crash before training even starts.
    """
    name = task.partition("_")[2]
    try:
        import gymnasium as gym

        env = gym.make(name)
        try:
            shape = getattr(env.observation_space, "shape", None)
            is_image = shape is not None and len(shape) == 3 and shape[-1] in (1, 3)
        finally:
            env.close()
        return "image" if is_image else "vector"
    except Exception:
        return "vector"


def build_config(task, preset, logdir, *, overrides=None):
    """Build a fully-resolved DreamerV3 config for ``task``.

    Layers, in order: defaults -> size preset -> per-suite defaults
    (CNN/MLP keys + train_ratio) -> ``logdir`` -> caller ``overrides``.
    """
    import dreamerv3
    from dreamerv3 import embodied

    if preset not in dreamerv3.Agent.configs:
        raise ValueError(
            f"Unknown preset {preset!r}. "
            f"Known: {sorted(dreamerv3.Agent.configs)}"
        )

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs[preset])

    suite = task.split("_", 1)[0]
    suite_defaults = dict(_SUITE_DEFAULTS.get(suite, {}))
    if suite == "gym":
        kind = _detect_gym_obs_kind(task)
        suite_defaults.update(
            _GYM_PIXEL_OVERRIDES if kind == "image" else _GYM_VECTOR_OVERRIDES
        )
    config = config.update(suite_defaults)
    config = config.update({"logdir": str(logdir)})

    if overrides:
        config = config.update(dict(overrides))

    return config


class Trainer:
    """Build a DreamerV3 agent + env + replay and run the training loop.

    Parameters
    ----------
    task : str
        Task spec of the form ``"<suite>_<name>"`` (e.g. ``"atari_pong"``).
    preset : str
        DreamerV3 size preset (``"size1m"``, ``"size12m"``, ...).
    logdir : str | pathlib.Path
        Directory where checkpoints, replay, and TensorBoard logs land.
    overrides : Mapping[str, object], optional
        Extra config keys applied last (e.g. ``{"run.steps": 20_000}``).
        Use this to bump steps, batch size, or any DreamerV3 config knob
        without dropping into raw flag strings.
    config : dreamerv3.embodied.Config, optional
        Pre-built config. The CLI passes this so user-supplied
        ``embodied.Flags`` overrides flow through. When omitted, the
        config is built from ``preset`` + per-suite defaults +
        ``overrides``.
    """

    def __init__(self, task, preset, logdir, *, overrides=None, config=None):
        from dreamerv3 import embodied

        if config is None:
            config = build_config(task, preset, logdir, overrides=overrides)

        self.task = task
        self.preset = preset
        self.config = config
        self.logdir = embodied.Path(config.logdir)

    def run(self):
        """Wire up agent + env + replay + logger and run the train loop."""
        import dreamerv3
        from dreamerv3 import embodied

        make_env = _import_make_env()

        self.logdir.mkdir()
        step = embodied.Counter()
        logger = embodied.Logger(
            step,
            [
                embodied.logger.TerminalOutput(),
                embodied.logger.JSONLOutput(self.logdir, "metrics.jsonl"),
                embodied.logger.TensorBoardOutput(self.logdir),
            ],
        )

        env = embodied.BatchEnv(
            [make_env(self.task, self.config)], parallel=False
        )
        agent = dreamerv3.Agent(env.obs_space, env.act_space, self.config)
        replay = embodied.replay.Replay(
            length=self.config.batch_length,
            capacity=self.config.replay.size,
            directory=self.logdir / "replay",
        )

        run_args = embodied.Config(
            **self.config.run,
            logdir=self.config.logdir,
            batch_size=self.config.batch_size,
            batch_length=self.config.batch_length,
        )
        embodied.run.train(agent, env, replay, logger, run_args)


def run_with_env_factory(config, make_env):
    """Run training given a fully-built config and a ``make_env(index)``
    factory.

    This is the entry point used by the per-suite tutorial scripts in
    ``examples/0X_*.py`` — they each build their own factory (so users
    can read a single self-contained file) but share the agent + replay
    + driver wiring with the ``Trainer`` class.
    """
    import dreamerv3
    from dreamerv3 import embodied

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

    env = embodied.BatchEnv([make_env(0)], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    replay = embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=logdir / "replay",
    )

    run_args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    embodied.run.train(agent, env, replay, logger, run_args)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--preset", default="size12m", type=str)
    parser.add_argument("--logdir", default="~/logdir/dreamerv3", type=str)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    from dreamerv3 import embodied

    config = build_config(args.task, args.preset, args.logdir)

    sys.argv = [sys.argv[0]] + remaining
    config = embodied.Flags(config).parse()

    Trainer(args.task, args.preset, args.logdir, config=config).run()


if __name__ == "__main__":
    main()
