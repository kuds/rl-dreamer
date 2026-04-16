"""Shared environment builders for the scripts in this directory.

``scripts/train.py``, ``scripts/evaluate.py``, ``scripts/record.py`` and
``scripts/visualize_dreams.py`` all need to construct the same env given a
``<suite>_<name>`` task spec. Keeping that logic in one place means the
per-suite hyperparameters (frame size, Atari action remapping, Minecraft
physics, ...) stay in sync across training and evaluation.
"""

from __future__ import annotations


def make_env(task: str, config, *, render_mode=None, return_info: bool = False):
    """Build and wrap the DreamerV3 env for ``task``.

    Parameters
    ----------
    task : str
        Task spec of the form ``"<suite>_<name>"`` (e.g. ``"atari_pong"``).
    config : dreamerv3.embodied.Config
        Config used by :func:`dreamerv3.wrap_env`.
    render_mode : str | None
        Forwarded to :func:`gymnasium.make` for suites that accept it
        (``gym``, ``minigrid``). ``scripts/record.py`` passes
        ``"rgb_array"`` so it can call ``raw_env.render()`` on vector-obs
        Gym envs.
    return_info : bool
        If True, also return a dict with ``raw_env`` (the underlying
        Gymnasium env, if any) and ``obs_key`` (``"image"`` or
        ``"vector"``). Callers that need ``.render()`` access use this.
    """
    import dreamerv3

    suite, _, name = task.partition("_")
    raw_env = None
    obs_key = "image"

    if suite == "gym":
        import gymnasium as gym
        from dreamerv3.embodied.envs import from_gym

        gym_kwargs = {}
        if render_mode is not None:
            gym_kwargs["render_mode"] = render_mode
        raw_env = gym.make(name, **gym_kwargs)

        # Heuristic: if the obs space is a Box of images, call it 'image',
        # otherwise 'vector'. Users can override via --enc.simple.*_keys.
        shape = getattr(raw_env.observation_space, "shape", None)
        is_image = shape is not None and len(shape) == 3 and shape[-1] in (1, 3)
        obs_key = "image" if is_image else "vector"
        env = from_gym.FromGym(raw_env, obs_key=obs_key)

    elif suite == "dmc":
        from dreamerv3.embodied.envs import dmc

        env = dmc.DMC(name, repeat=2, size=(64, 64))

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

    elif suite == "crafter":
        import crafter
        from dreamerv3.embodied.envs import from_gym

        env = crafter.Env()
        env = from_gym.FromGym(env, obs_key="image")

    elif suite == "minigrid":
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
        from dreamerv3.embodied.envs import from_gym

        gym_kwargs = {}
        if render_mode is not None:
            gym_kwargs["render_mode"] = render_mode
        raw_env = gym.make(name, **gym_kwargs)
        mg_env = RGBImgPartialObsWrapper(raw_env, tile_size=8)
        mg_env = ImgObsWrapper(mg_env)
        env = from_gym.FromGym(mg_env, obs_key="image")

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

    else:
        raise ValueError(f"Unknown task suite: {suite!r} (from task={task!r})")

    env = dreamerv3.wrap_env(env, config)

    if return_info:
        return env, {"raw_env": raw_env, "obs_key": obs_key}
    return env
