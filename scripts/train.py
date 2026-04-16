"""Generic DreamerV3 training entry point.

Picks an environment suite based on a `--task` prefix and otherwise forwards
all command-line flags to DreamerV3's config system.

Examples:
    python scripts/train.py --task gym_CartPole-v1 --preset size1m
    python scripts/train.py --task dmc_walker_walk --preset size12m
    python scripts/train.py --task atari_pong --preset size25m
    python scripts/train.py --task crafter_reward --preset size25m
    python scripts/train.py --task minigrid_MiniGrid-DoorKey-6x6-v0
    python scripts/train.py --task minecraft_diamond --preset size25m
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")
warnings.filterwarnings("ignore", ".*the imp module is deprecated.*")


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--preset", default="size12m", type=str)
    parser.add_argument("--logdir", default="~/logdir/dreamerv3", type=str)
    # Everything else is forwarded to embodied.Flags.
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    import dreamerv3
    from dreamerv3 import embodied

    from env_builders import make_env

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    if args.preset not in dreamerv3.Agent.configs:
        raise ValueError(
            f"Unknown preset {args.preset!r}. Known: "
            f"{sorted(dreamerv3.Agent.configs)}"
        )
    config = config.update(dreamerv3.Agent.configs[args.preset])

    # Task-specific defaults.
    suite = args.task.split("_", 1)[0]
    task_defaults = {
        "gym": {"run.train_ratio": 32},
        "dmc": {"run.train_ratio": 512},
        "atari": {"run.train_ratio": 64},
        "crafter": {"run.train_ratio": 64},
        "minigrid": {"run.train_ratio": 32},
        "minecraft": {"run.train_ratio": 16},
    }
    config = config.update(task_defaults.get(suite, {}))
    config = config.update({"logdir": args.logdir})

    # Forward anything else on the command line to the embodied Flag parser.
    sys.argv = [sys.argv[0]] + remaining
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

    env = embodied.BatchEnv([make_env(args.task, config)], parallel=False)
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


if __name__ == "__main__":
    main()
