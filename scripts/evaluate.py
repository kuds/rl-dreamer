"""Load a trained DreamerV3 checkpoint and roll out evaluation episodes.

Example:
    python scripts/evaluate.py \
        --task gym_CartPole-v1 \
        --logdir ~/logdir/cartpole \
        --episodes 10
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
    parser.add_argument("--logdir", required=True, type=str)
    parser.add_argument("--episodes", default=10, type=int)
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    import numpy as np

    import dreamerv3
    from dreamerv3 import embodied

    # Import the same env builder used for training.
    sys.path.insert(0, str(embodied.Path(__file__).parent))
    from train import make_env  # type: ignore

    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(dreamerv3.Agent.configs[args.preset])
    config = config.update({"logdir": args.logdir})
    sys.argv = [sys.argv[0]] + remaining
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    env = embodied.BatchEnv([make_env(args.task, config)], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)

    # Restore the most recent checkpoint.
    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.agent = agent
    checkpoint.load(keys=["agent"])

    returns = []
    for ep in range(args.episodes):
        obs = env.reset()
        state = None
        total = 0.0
        done = False
        while not done:
            action, state = agent.policy(obs, state, mode="eval")
            obs = env.step(action)
            total += float(obs["reward"][0])
            done = bool(obs["is_last"][0])
        returns.append(total)
        print(f"episode {ep + 1}/{args.episodes}: return={total:.2f}")

    returns = np.array(returns)
    print(
        f"\nMean return: {returns.mean():.3f} +/- {returns.std():.3f} "
        f"(n={len(returns)})"
    )


if __name__ == "__main__":
    main()
