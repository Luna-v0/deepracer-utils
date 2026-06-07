"""Reliable RL evaluation for DeepRacer runs, via `rliable`.

Implements the evaluation protocol from *"Deep Reinforcement Learning at the
Edge of the Statistical Precipice"* (Agarwal et al., NeurIPS 2021) on top of
the dr-gym Tier-1 trace. Point estimates of the mean hide enormous run-to-run
variance in RL; this module instead reports **interval estimates** with
stratified-bootstrap confidence intervals, and the companion tools that don't
collapse a whole training campaign to one fragile number:

- **Aggregate metrics** — Median, **IQM** (inter-quartile mean: drops the top &
  bottom 25% of runs, the recommended robust aggregate), Mean, and
  **Optimality Gap** (how far from "solved"), each with 95% CIs.
- **Performance profiles** — the fraction of runs scoring above every threshold
  τ; a curve that stochastically dominates is unambiguously better.
- **Probability of improvement** — P(method X beats method Y) per pair, with CIs.
- **Sample-efficiency curve** — IQM of the metric over training, with CIs.

`rliable` is an *optional* dependency (imported lazily inside each function),
so importing :mod:`deepracer.logs` never requires it. Install with
``pip install deepracer-utils[rliable]`` (or ``pip install rliable``).

Scores
------
The "score" matrix has shape ``(num_runs, num_samples)`` per method, where a
*run* is one training run (Optuna trial / seed) and *samples* are that run's
per-episode metric values (evaluation episodes when available). ``progress`` is
normalised to ``[0, 1]`` (÷100) so the optimality gap is meaningful; pass
``normalize=False`` or a ``max_score`` for unbounded metrics like reward.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

RunInput = Union[str, pd.DataFrame]


# --------------------------------------------------------------------------- #
# Building score matrices from traces
# --------------------------------------------------------------------------- #

def _as_df(run: RunInput) -> pd.DataFrame:
    if isinstance(run, pd.DataFrame):
        return run
    from .gym_trace import load_gym_trace

    return load_gym_trace(run)


def per_episode_scores(
    df: pd.DataFrame,
    metric: str = "progress",
    phase: Optional[str] = None,
    normalize: bool = True,
    max_score: Optional[float] = None,
) -> np.ndarray:
    """Per-episode scores for one run.

    Args:
        df: a loaded gym trace.
        metric: ``"progress"`` (max % per episode), ``"reward"`` /
            ``"eval_reward"`` (sum per episode), or any per-step column (mean).
        phase: keep only episodes of this phase (``"eval"``/``"train"``) when
            the ``phase`` column is present; ``None`` keeps all.
        normalize: for ``progress``, divide by 100 → ``[0, 1]``.
        max_score: if given, divide scores by this (for unbounded metrics).
    """
    if phase is not None and "phase" in df.columns and (df["phase"] == phase).any():
        df = df[df["phase"] == phase]
    col = "unique_episode" if "unique_episode" in df.columns else "episode"
    g = df.groupby(col)
    if metric == "progress":
        s = g["progress"].max()
    elif metric in ("reward", "eval_reward"):
        s = g[metric].sum()
    else:
        s = g[metric].mean()
    scores = s.to_numpy(dtype=np.float64)
    if max_score is not None:
        scores = scores / float(max_score)
    elif normalize and metric == "progress":
        scores = scores / 100.0
    return scores


def score_matrix(
    runs: Sequence[RunInput],
    metric: str = "progress",
    phase: Optional[str] = "eval",
    n_samples: Optional[int] = None,
    **kw,
) -> np.ndarray:
    """Stack per-run episode scores into a ``(num_runs, n_samples)`` matrix.

    Runs are truncated to a common ``n_samples`` (default: the shortest run's
    episode count for the chosen phase) so the matrix is rectangular, as
    rliable requires.
    """
    per_run = [per_episode_scores(_as_df(r), metric=metric, phase=phase, **kw) for r in runs]
    per_run = [s for s in per_run if s.size > 0]
    if not per_run:
        raise ValueError("no episodes found for the requested metric/phase")
    n = n_samples or min(s.size for s in per_run)
    # take the LAST n episodes of each run (most-trained behaviour)
    return np.vstack([s[-n:] for s in per_run])


def score_dict(
    methods: Dict[str, Sequence[RunInput]],
    metric: str = "progress",
    phase: Optional[str] = "eval",
    n_samples: Optional[int] = None,
    **kw,
) -> Dict[str, np.ndarray]:
    """Build ``{method_name: (num_runs, n_samples)}`` for several methods.

    A "method" is a configuration you want to compare (e.g. two reward
    functions, two algorithms); each maps to a list of its runs (seeds/trials).
    For a single configuration, pass one method — the aggregate metrics still
    apply (they characterise that config's run-to-run reliability).
    """
    n = n_samples
    if n is None:
        # common sample count across ALL runs of ALL methods
        lengths = []
        for runs in methods.values():
            for r in runs:
                s = per_episode_scores(_as_df(r), metric=metric, phase=phase, **kw)
                if s.size:
                    lengths.append(s.size)
        n = min(lengths) if lengths else 1
    return {name: score_matrix(runs, metric=metric, phase=phase, n_samples=n, **kw)
            for name, runs in methods.items()}


# --------------------------------------------------------------------------- #
# 1. Aggregate metrics with stratified-bootstrap CIs
# --------------------------------------------------------------------------- #

def aggregate_metrics(scores: Dict[str, np.ndarray], reps: int = 50000):
    """Median / IQM / Mean / Optimality-Gap point + interval estimates.

    Returns ``(point_estimates, interval_estimates)`` exactly as
    ``rliable.library.get_interval_estimates`` produces them: each is a dict
    keyed by method, with a length-4 vector / ``(2, 4)`` CI array in the order
    ``[Median, IQM, Mean, Optimality Gap]``.
    """
    from rliable import library as rly
    from rliable import metrics

    def _agg(x: np.ndarray) -> np.ndarray:
        return np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ])

    return rly.get_interval_estimates(scores, _agg, reps=reps)


def plot_aggregate_metrics(scores: Dict[str, np.ndarray], reps: int = 50000):
    """Plot the four aggregate metrics with 95% CIs. Returns ``(fig, axes)``."""
    from rliable import plot_utils

    point, interval = aggregate_metrics(scores, reps=reps)
    fig, axes = plot_utils.plot_interval_estimates(
        point, interval,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=list(scores.keys()),
        xlabel="normalised score",
    )
    return fig, axes


# --------------------------------------------------------------------------- #
# 2. Performance profiles
# --------------------------------------------------------------------------- #

def performance_profile(scores: Dict[str, np.ndarray],
                        taus: Optional[np.ndarray] = None, reps: int = 2000):
    """Run-score distributions ``P(score > τ)`` over thresholds ``taus``.

    Returns ``(taus, distributions, distribution_cis)``. A profile that lies
    above another everywhere means that method is better at *every* threshold —
    a far stronger statement than a single mean.
    """
    from rliable import library as rly

    if taus is None:
        hi = max(float(np.max(s)) for s in scores.values())
        taus = np.linspace(0.0, max(hi, 1e-6), 100)
    dists, cis = rly.create_performance_profile(scores, taus, reps=reps)
    return taus, dists, cis


def plot_performance_profile(scores: Dict[str, np.ndarray],
                             taus: Optional[np.ndarray] = None, reps: int = 2000):
    """Plot performance profiles with CIs. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from rliable import plot_utils

    taus, dists, cis = performance_profile(scores, taus, reps=reps)
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        dists, taus, performance_profile_cis=cis,
        colors=dict(zip(scores.keys(), plt.cm.tab10.colors)),
        xlabel=r"normalised score $(\tau)$", ax=ax,
    )
    ax.set_ylabel(r"fraction of runs with score $> \tau$")
    return fig, ax


# --------------------------------------------------------------------------- #
# 3. Probability of improvement
# --------------------------------------------------------------------------- #

def probability_of_improvement(scores: Dict[str, np.ndarray],
                               pairs: Sequence[Tuple[str, str]], reps: int = 1000):
    """P(method X > method Y) for each ``(X, Y)`` pair, with CIs.

    Returns ``(point_estimates, interval_estimates)`` keyed by ``"X,Y"``.
    """
    from rliable import library as rly
    from rliable import metrics

    pair_scores = {f"{x},{y}": (scores[x], scores[y]) for x, y in pairs}
    return rly.get_interval_estimates(
        pair_scores, metrics.probability_of_improvement, reps=reps,
    )


def plot_probability_of_improvement(scores: Dict[str, np.ndarray],
                                    pairs: Sequence[Tuple[str, str]], reps: int = 1000):
    """Plot P(X > Y) per pair with CIs. Returns ``(fig, ax)``."""
    from rliable import plot_utils

    point, interval = probability_of_improvement(scores, pairs, reps=reps)
    ax = plot_utils.plot_probability_of_improvement(point, interval)
    return ax.get_figure(), ax


# --------------------------------------------------------------------------- #
# 4. Sample-efficiency curve
# --------------------------------------------------------------------------- #

def sample_efficiency(
    runs: Sequence[RunInput],
    metric: str = "progress",
    phase: Optional[str] = None,
    n_bins: int = 10,
    reps: int = 2000,
    **kw,
):
    """IQM of *metric* over training, with 95% CIs — a learning curve done right.

    Each run's episodes are split into ``n_bins`` ordered bins (early→late
    training); the IQM across runs is bootstrapped per bin. Returns
    ``(x, iqm, cis)`` where ``x`` is the bin's mean episode fraction (0–1),
    ``iqm`` is ``(n_bins,)`` and ``cis`` is ``(2, n_bins)``.
    """
    from rliable import library as rly
    from rliable import metrics

    binned = []
    for r in runs:
        s = per_episode_scores(_as_df(r), metric=metric, phase=phase, **kw)
        if s.size < n_bins:
            continue
        idx = np.array_split(np.arange(s.size), n_bins)
        binned.append([float(np.mean(s[i])) for i in idx])
    if not binned:
        raise ValueError("not enough episodes per run for the requested n_bins")
    mat = np.array(binned)  # (num_runs, n_bins)

    # rliable expects {method: (runs, tasks, frames)}; one task here.
    frames_scores = {"runs": mat[:, None, :]}
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x[..., k]) for k in range(x.shape[-1])])
    iqm, cis = rly.get_interval_estimates(frames_scores, iqm_fn, reps=reps)
    x = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins
    return x, iqm["runs"], cis["runs"]


def plot_sample_efficiency(
    runs: Sequence[RunInput],
    metric: str = "progress",
    phase: Optional[str] = None,
    n_bins: int = 10,
    reps: int = 2000,
    label: str = "runs",
    **kw,
):
    """Plot the sample-efficiency curve with a CI band. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from rliable import plot_utils

    x, iqm, cis = sample_efficiency(runs, metric=metric, phase=phase,
                                    n_bins=n_bins, reps=reps, **kw)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_utils.plot_sample_efficiency_curve(
        x, {label: iqm}, {label: cis},
        xlabel="training progress (episode fraction)",
        ylabel=f"IQM {metric}", ax=ax,
    )
    return fig, ax
