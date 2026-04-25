"""Shared training-loop glue for examples/01..07.

Each example builds its own config (the interesting, task-specific
part) and its own ``make_env`` factory. The logger + batched env +
agent + replay + driver wiring is pure boilerplate, so it lives here
to avoid duplication.
"""

from __future__ import annotations


def run_training(config, make_env):
    """Wire up the standard DreamerV3 training loop and run it.

    Parameters
    ----------
    config : dreamerv3.embodied.Config
        Fully-resolved config (defaults + preset + task overrides + any
        CLI flag overrides). The caller is responsible for producing
        this.
    make_env : Callable[[int], dreamerv3.embodied.Env]
        Factory returning a single DreamerV3-ready env for worker
        ``index``. The returned env should already have been wrapped
        via ``dreamerv3.wrap_env(env, config)``.
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

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
    )
    embodied.run.train(agent, env, replay, logger, args)
