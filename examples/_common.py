"""Shared training-loop glue for examples/01..07.

Each example builds its own config (the interesting, task-specific part)
and its own ``make_env`` factory. The agent + replay + driver wiring is
pure boilerplate, so we delegate to ``scripts.train.run_with_env_factory``
to keep a single source of truth shared with the generic
``scripts/train.py`` entry point.
"""

from __future__ import annotations

import os
import sys


def _add_repo_root_to_path():
    """Make ``scripts/`` importable when the example is run as a script
    from the repo root (``python examples/01_gym_cartpole.py``)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def run_training(config, make_env):
    """Run the standard DreamerV3 training loop.

    Parameters
    ----------
    config : dreamerv3.embodied.Config
        Fully-resolved config (defaults + preset + task overrides + any
        CLI flag overrides).
    make_env : Callable[[int], dreamerv3.embodied.Env]
        Factory returning a single DreamerV3-ready env for worker
        ``index``. The returned env should already have been wrapped via
        ``dreamerv3.wrap_env(env, config)``.
    """
    _add_repo_root_to_path()
    from scripts.train import run_with_env_factory

    run_with_env_factory(config, make_env)
