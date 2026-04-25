"""Render the DreamerV3 network as static architecture diagrams.

This module is both a CLI script and a library. Notebooks and other
callers can import the render helpers directly::

    from scripts.visualize_network import render_diagram, render_all

    fig, _ = render_diagram("world_model")        # inline in a notebook
    render_all(output_dir="docs/figures")         # write all PNGs to disk

Three diagrams are defined:

1. ``world_model``  — the Recurrent State-Space Model (RSSM) plus
   encoder, decoder, and the reward / continue prediction heads.
2. ``imagination``  — the open-loop imagination rollout that feeds
   the actor-critic learning loop.
3. ``pipeline``     — the outer three-loop training diagram
   (environment → replay → world model → imagination → actor/critic).

The diagrams are schematic. Tensor shapes are shown symbolically
(``D``, ``S``, ``N``) rather than with preset-specific numbers, since
those drift across DreamerV3 releases.

CLI usage is preserved::

    # Render all three diagrams into docs/figures/
    python scripts/visualize_network.py

    # Pick a different output directory
    python scripts/visualize_network.py --output /tmp/dreamer_figures

    # Render only one diagram
    python scripts/visualize_network.py --only world_model

Requirements beyond the base install:
    pip install matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

# Color palette — grouped by architectural role. Chosen for clarity on
# both light and dark backgrounds.
PALETTE = {
    "env": "#d9e7ff",          # real environment
    "replay": "#cfe8d4",       # replay buffer
    "encoder": "#ffe0b3",      # observation encoder
    "decoder": "#ffe0b3",      # observation decoder
    "rssm_det": "#c7d9f5",     # RSSM deterministic (GRU / h_t)
    "rssm_stoch": "#f5c9c9",   # RSSM stochastic (z_t)
    "head": "#e8d6f5",         # prediction heads (reward / continue)
    "actor": "#fce3a8",        # policy
    "critic": "#b8e0d2",       # value
    "loss": "#f2f2f2",         # losses
    "arrow": "#555555",
    "edge": "#333333",
    "text": "#1a1a1a",
    "subtitle": "#555555",
}

BOX_EDGE_WIDTH = 1.4
ARROW_LW = 1.4


def _box(
    ax,
    xy,
    width,
    height,
    label,
    color,
    *,
    shape_text=None,
    rounded=True,
    fontsize=10,
    fontweight="bold",
):
    """Draw a labelled rectangle centred at ``xy``."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    x, y = xy
    left = x - width / 2
    bottom = y - height / 2
    if rounded:
        patch = FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=BOX_EDGE_WIDTH,
            edgecolor=PALETTE["edge"],
            facecolor=color,
        )
    else:
        patch = Rectangle(
            (left, bottom),
            width,
            height,
            linewidth=BOX_EDGE_WIDTH,
            edgecolor=PALETTE["edge"],
            facecolor=color,
        )
    ax.add_patch(patch)

    if shape_text:
        ax.text(
            x,
            y + 0.07,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=PALETTE["text"],
        )
        ax.text(
            x,
            y - 0.12,
            shape_text,
            ha="center",
            va="center",
            fontsize=fontsize - 2,
            color=PALETTE["subtitle"],
            style="italic",
        )
    else:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=PALETTE["text"],
        )


def _arrow(ax, start, end, *, label=None, style="->", curve=0.0, ls="-"):
    """Draw a (possibly curved) arrow between two points."""
    from matplotlib.patches import FancyArrowPatch

    connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=ARROW_LW,
        color=PALETTE["arrow"],
        connectionstyle=connectionstyle,
        linestyle=ls,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["subtitle"],
            style="italic",
        )


def _title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=10,
            color=PALETTE["subtitle"],
            style="italic",
        )


def _frame(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------
# Diagram 1 — World model (RSSM)
# --------------------------------------------------------------------------


def draw_world_model(ax):
    """RSSM world model: encoder + recurrent dynamics + heads."""
    _frame(ax, xlim=(0, 12), ylim=(0, 7))
    _title(
        ax,
        "DreamerV3 World Model (RSSM)",
        subtitle="Encoder + recurrent dynamics + prediction heads",
    )

    # Inputs.
    _box(ax, (1.0, 5.5), 1.6, 0.7, "obs  $o_t$", PALETTE["env"])
    _box(ax, (1.0, 3.3), 1.6, 0.7, "action  $a_{t-1}$", PALETTE["env"])
    _box(ax, (1.0, 1.1), 1.6, 0.7, "prev state\n$h_{t-1},\\, z_{t-1}$", PALETTE["rssm_det"])

    # Encoder.
    _box(
        ax,
        (3.4, 5.5),
        1.6,
        0.8,
        "Encoder",
        PALETTE["encoder"],
        shape_text="CNN + MLP → $x_t$",
    )

    # GRU (deterministic).
    _box(
        ax,
        (5.8, 2.2),
        1.8,
        0.9,
        "GRU",
        PALETTE["rssm_det"],
        shape_text="deterministic $h_t$",
    )

    # Prior and posterior stochastic heads.
    _box(
        ax,
        (8.2, 3.3),
        2.0,
        0.8,
        "Prior  $p(z_t \\mid h_t)$",
        PALETTE["rssm_stoch"],
        shape_text="MLP → logits",
    )
    _box(
        ax,
        (8.2, 5.5),
        2.0,
        0.8,
        "Posterior  $q(z_t \\mid h_t, x_t)$",
        PALETTE["rssm_stoch"],
        shape_text="MLP → logits",
    )

    # KL bridge between prior and posterior.
    ax.annotate(
        "",
        xy=(8.2, 4.95),
        xytext=(8.2, 3.75),
        arrowprops=dict(arrowstyle="<->", linestyle="--", color=PALETTE["arrow"]),
    )
    ax.text(
        8.55,
        4.35,
        "KL",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["subtitle"],
    )

    # Prediction heads.
    _box(ax, (10.8, 6.1), 1.8, 0.7, "Decoder  $\\hat o_t$", PALETTE["decoder"])
    _box(ax, (10.8, 4.9), 1.8, 0.7, "Reward  $\\hat r_t$", PALETTE["head"])
    _box(ax, (10.8, 3.7), 1.8, 0.7, "Continue  $\\hat \\gamma_t$", PALETTE["head"])

    # Arrows: obs → encoder
    _arrow(ax, (1.8, 5.5), (2.6, 5.5))
    # encoder → posterior
    _arrow(ax, (4.2, 5.5), (7.2, 5.5))
    # prev state → GRU
    _arrow(ax, (1.8, 1.1), (4.9, 2.0), curve=0.15)
    # action → GRU
    _arrow(ax, (1.8, 3.3), (4.9, 2.4), curve=-0.15)
    # GRU → h_t → prior
    _arrow(ax, (6.7, 2.4), (7.2, 3.15))
    # GRU → h_t → posterior (context for q)
    _arrow(ax, (6.7, 2.55), (7.2, 5.3), curve=-0.25)
    # posterior → decoder / reward / continue
    _arrow(ax, (9.2, 5.6), (9.9, 6.1), curve=0.0)
    _arrow(ax, (9.2, 5.3), (9.9, 4.95), curve=0.0)
    _arrow(ax, (9.2, 5.1), (9.9, 3.75), curve=0.1)

    # Legend for prior vs posterior.
    ax.text(
        0.25,
        0.3,
        "During training: posterior + prior, KL regularized.\n"
        "During imagination: prior only (no real obs).",
        fontsize=9,
        color=PALETTE["subtitle"],
        style="italic",
    )


# --------------------------------------------------------------------------
# Diagram 2 — Imagination rollout
# --------------------------------------------------------------------------


def draw_imagination(ax):
    """Imagined rollout through the prior, scored by actor / critic."""
    _frame(ax, xlim=(0, 12), ylim=(0, 7))
    _title(
        ax,
        "Imagination Rollout + Actor-Critic",
        subtitle="Policy is trained in latent space, never touches pixels",
    )

    # Seed latent.
    _box(
        ax,
        (1.1, 3.5),
        1.8,
        0.9,
        "seed latent\n$s_0$ from replay",
        PALETTE["rssm_stoch"],
    )

    # Latent chain.
    xs = [3.3, 5.3, 7.3, 9.3]
    for i, x in enumerate(xs):
        _box(ax, (x, 3.5), 1.2, 0.8, f"$s_{i}$", PALETTE["rssm_det"])

    # Actors above.
    for i, x in enumerate(xs[:-1]):
        ax_x = (x + xs[i + 1]) / 2
        _box(ax, (ax_x, 5.2), 1.2, 0.6, "Actor $\\pi$", PALETTE["actor"], fontsize=9)
        # s_i -> pi
        _arrow(ax, (x, 3.9), (ax_x - 0.25, 4.9), curve=0.15)
        # pi -> s_{i+1}  (action a_i delivered to next prior step)
        _arrow(ax, (ax_x + 0.25, 4.9), (xs[i + 1], 3.9), curve=0.15)
        ax.text(
            ax_x + 0.65,
            4.45,
            f"$a_{i}$",
            ha="left",
            va="center",
            fontsize=10,
            color=PALETTE["subtitle"],
            style="italic",
        )
        # Prior step: s_i -> s_{i+1}  (dynamics in latent space)
        _arrow(ax, (x + 0.6, 3.5), (xs[i + 1] - 0.6, 3.5))

    # Critics below.
    for i, x in enumerate(xs):
        _box(ax, (x, 1.8), 1.2, 0.6, "Critic $V$", PALETTE["critic"], fontsize=9)
        _arrow(ax, (x, 3.1), (x, 2.1), curve=0.0)

    # Returns box.
    _box(
        ax,
        (10.9, 1.8),
        1.8,
        0.9,
        "$\\lambda$-returns\n+ losses",
        PALETTE["loss"],
    )
    _arrow(ax, (9.9, 1.8), (10.0, 1.8))
    # Feed all critic outputs into returns box via a faint bus line.
    ax.plot(
        [xs[0], xs[-1]],
        [1.15, 1.15],
        color=PALETTE["arrow"],
        linewidth=0.8,
        linestyle=":",
    )

    # seed -> s_0
    _arrow(ax, (2.0, 3.5), (2.7, 3.5))

    # Gradient flow annotation.
    ax.text(
        6.2,
        0.6,
        "Gradients flow back through the world model dynamics —"
        " the actor learns *through* the learned model, not from"
        " scalar rewards alone.",
        ha="center",
        fontsize=9,
        color=PALETTE["subtitle"],
        style="italic",
    )


# --------------------------------------------------------------------------
# Diagram 3 — Full pipeline (three nested loops)
# --------------------------------------------------------------------------


def draw_pipeline(ax):
    """End-to-end DreamerV3 training loop."""
    _frame(ax, xlim=(0, 12), ylim=(-0.5, 6))
    _title(
        ax,
        "DreamerV3 Training Pipeline",
        subtitle="Three nested loops: env interaction → world model → actor-critic",
    )

    env = (1.4, 3.0)
    replay = (4.0, 3.0)
    wm = (6.6, 4.2)
    imag = (6.6, 1.8)
    ac = (9.2, 3.0)

    _box(ax, env, 1.8, 0.9, "Environment", PALETTE["env"])
    _box(ax, replay, 1.8, 0.9, "Replay\nBuffer", PALETTE["replay"])
    _box(ax, wm, 2.0, 0.9, "World Model\nLearning", PALETTE["rssm_det"])
    _box(ax, imag, 2.0, 0.9, "Imagined\nRollouts", PALETTE["rssm_stoch"])
    _box(ax, ac, 1.8, 0.9, "Actor + Critic\nLearning", PALETTE["actor"])

    _arrow(ax, (env[0] + 0.9, env[1]), (replay[0] - 0.9, replay[1]), label="transitions")
    _arrow(
        ax,
        (replay[0] + 0.9, replay[1] + 0.1),
        (wm[0] - 1.0, wm[1] - 0.15),
        label="batches",
        curve=0.1,
    )
    _arrow(
        ax,
        (wm[0] + 0.6, wm[1] - 0.45),
        (imag[0] + 0.6, imag[1] + 0.45),
        label="latent $s_0$",
    )
    _arrow(
        ax,
        (imag[0] + 1.0, imag[1] + 0.1),
        (ac[0] - 0.9, ac[1] - 0.15),
        label="$\\lambda$-returns",
        curve=-0.1,
    )
    # Actor-critic -> policy -> environment (feedback).
    _arrow(
        ax,
        (ac[0], ac[1] - 0.45),
        (env[0], env[1] - 0.45),
        curve=-0.4,
    )
    ax.text(
        (env[0] + ac[0]) / 2,
        0.55,
        "policy $\\pi$",
        ha="center",
        va="center",
        fontsize=9,
        color=PALETTE["subtitle"],
        style="italic",
    )

    # Side note on train_ratio.
    ax.text(
        6.0,
        -0.1,
        "Ratio of env steps to gradient updates is set by run.train_ratio.",
        ha="center",
        fontsize=9,
        color=PALETTE["subtitle"],
        style="italic",
        family="monospace",
    )


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------


DIAGRAMS = {
    "world_model": ("world_model.png", draw_world_model, (12, 7)),
    "imagination": ("imagination.png", draw_imagination, (12, 7)),
    "pipeline": ("pipeline.png", draw_pipeline, (12, 6)),
}


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "scripts/visualize_network.py needs matplotlib. Install it with:\n"
            "    pip install matplotlib"
        ) from exc


def render_diagram(name, ax=None, *, figsize=None):
    """Render a single named diagram.

    Parameters
    ----------
    name : str
        One of the keys in :data:`DIAGRAMS` (``"world_model"``,
        ``"imagination"``, ``"pipeline"``).
    ax : matplotlib.axes.Axes, optional
        Render onto this Axes. When ``None``, a fresh figure is created
        with the diagram's default ``figsize``.
    figsize : tuple, optional
        Override the default figsize when creating a new figure. Ignored
        when ``ax`` is provided.

    Returns
    -------
    (fig, ax) : tuple
        The matplotlib Figure and Axes containing the diagram.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if name not in DIAGRAMS:
        raise ValueError(
            f"Unknown diagram {name!r}. Known: {sorted(DIAGRAMS)}"
        )
    _filename, draw_fn, default_figsize = DIAGRAMS[name]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or default_figsize)
    else:
        fig = ax.figure
    draw_fn(ax)
    fig.tight_layout()
    return fig, ax


def render_all(output_dir="docs/figures", *, dpi=160, only=None, verbose=True):
    """Render diagrams as PNG files.

    Parameters
    ----------
    output_dir : str | pathlib.Path
        Directory to write PNGs into. Created if missing.
    dpi : int
        Output resolution.
    only : str, optional
        Render just one diagram by name.
    verbose : bool
        Print one line per file written.

    Returns
    -------
    list of pathlib.Path
        Paths of files written, in render order.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [only] if only else list(DIAGRAMS.keys())
    written = []
    for name in selected:
        if name not in DIAGRAMS:
            raise ValueError(
                f"Unknown diagram {name!r}. Known: {sorted(DIAGRAMS)}"
            )
        filename, _draw_fn, _figsize = DIAGRAMS[name]
        fig, _ax = render_diagram(name)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        if verbose:
            print(f"wrote {out_path}")
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render DreamerV3 network architecture diagrams."
    )
    parser.add_argument(
        "--output",
        default="docs/figures",
        type=str,
        help="Directory to write PNG files into. Created if missing.",
    )
    parser.add_argument(
        "--only",
        default=None,
        choices=sorted(DIAGRAMS.keys()),
        help="Render only one diagram instead of all three.",
    )
    parser.add_argument(
        "--dpi",
        default=160,
        type=int,
        help="Output resolution in DPI (default: 160).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    _require_matplotlib()
    import matplotlib

    # The CLI writes PNGs to disk — no display needed — so force the
    # non-interactive backend. Library callers (notebooks, etc.) manage
    # their own backend.
    matplotlib.use("Agg")

    written = render_all(output_dir=args.output, dpi=args.dpi, only=args.only)
    print(f"\n{len(written)} diagram(s) written to {Path(args.output).expanduser()}")


if __name__ == "__main__":
    main()
