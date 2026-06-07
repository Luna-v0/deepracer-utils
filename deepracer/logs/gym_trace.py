"""Loader + track-path plotting for the **dr-gym Tier-1 trace**.

dr-gym (the SB3 + MLflow + TensorBoard + Optuna training pipeline) writes a
per-step "Tier-1 trace" as Parquet shards — one file per episode under
``<run_dir>/trace/steps/ep_NNNNNN.parquet``. The column names were chosen to
match this library's *internal* DataFrame (``steering_angle``, ``speed``,
``on_track``, ``progress``, ``closest_waypoint``, ``track_len``,
``episode_status``, …), so once loaded the trace flows straight into the
existing analysis (``AnalysisUtils``, ``SimtraceStabilityAnalyzer``,
``PlottingUtils``) — **no S3, no robomaker logs, no rigid folder contract.**

This module is the bridge:

- :func:`load_gym_trace` / :class:`GymTraceLog` — read the shards into an
  analysis-ready DataFrame, deriving the few columns the classic loaders add
  (``tstamp``, ``wall_clock``, ``worker``, ``iteration``, ``unique_episode``).
- :func:`plot_episode_path` / :func:`plot_last_eval_path` — draw the car's path
  on the track outline (the "where did it drive" view), using a Track loaded
  from a ``.npy`` waypoint file.

It deliberately does **not** import ``gym_dr``: the contract is the on-disk
Parquet format, not the producer. dr-gym owns the writer; this owns the reader.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd

# Columns the classic deepracer-utils DataFrame carries that the in-process
# gym trace does not store directly; we derive them on load.
_DERIVED = ("tstamp", "wall_clock", "worker", "iteration", "unique_episode", "pause_duration")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _resolve_shards(path: str) -> List[str]:
    """Find the Parquet shard files for *path*, leniently.

    Accepts (in priority order):
      * a single ``.parquet`` file,
      * a directory containing ``ep_*.parquet`` (the ``trace/steps`` dir),
      * a run directory containing ``trace/steps/ep_*.parquet``,
      * a glob pattern.
    """
    if path.endswith(".parquet") and os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        for sub in ("", "trace/steps", "steps"):
            cand = sorted(glob.glob(os.path.join(path, sub, "ep_*.parquet")))
            if cand:
                return cand
        # last resort: any parquet under the dir
        cand = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if cand:
            return cand
        raise FileNotFoundError(f"no trace shards (ep_*.parquet) found under {path!r}")
    # treat as a glob
    cand = sorted(glob.glob(path))
    if not cand:
        raise FileNotFoundError(f"no files match {path!r}")
    return cand


def load_gym_trace(
    path: str,
    run_name: Optional[str] = None,
    episode_offset: int = 0,
) -> pd.DataFrame:
    """Load a dr-gym Tier-1 trace into an analysis-ready DataFrame.

    Args:
        path: a run dir, a ``trace/steps`` dir, a single ``.parquet`` shard, or
            a glob — see :func:`_resolve_shards`.
        run_name: value for the added ``run`` column. Defaults to the basename
            of the run directory.
        episode_offset: added to ``unique_episode`` so several runs can be
            concatenated without episode-index collisions (see
            :func:`load_gym_traces`).

    Returns:
        A DataFrame carrying both the native gym-trace columns (``world_name``,
        ``chunk_index``, ``phase``, ``eval_reward``, OA fields, …) **and** the
        classic internal columns the rest of this library expects:

        * ``tstamp`` ← ``sim_time`` when present, else ``wall_time`` (the
          in-process producer has no simulator clock, so this is wall time;
          a bag-derived trace fills real ``sim_time``).
        * ``wall_clock`` ← ``wall_time``.
        * ``worker`` ← 0 (gym traces are single-worker).
        * ``iteration`` ← ``chunk_index`` (the runtime track-swap chunk — the
          closest analog to a DRfC training iteration; 0 for single-world runs).
        * ``unique_episode`` ← ``episode`` (already globally unique within a
          run) ``+ episode_offset``.
        * ``pause_duration`` ← NaN (not modelled in the gym env).

    Rows are sorted by ``unique_episode`` then ``steps`` — matching
    ``DeepRacerLog.load_training_trace``.
    """
    shards = _resolve_shards(path)
    df = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)

    if run_name is None:
        run_name = _infer_run_name(path)
    df["run"] = run_name

    # Clock: prefer the simulator clock; fall back to wall time.
    if "sim_time" in df.columns and df["sim_time"].notna().any():
        df["tstamp"] = df["sim_time"].where(df["sim_time"].notna(), df.get("wall_time"))
    else:
        df["tstamp"] = df.get("wall_time")
    df["wall_clock"] = df.get("wall_time")
    df["worker"] = 0
    df["iteration"] = df.get("chunk_index", 0)
    df["iteration"] = df["iteration"].fillna(0).astype(int)
    df["unique_episode"] = df["episode"].astype(int) + int(episode_offset)
    if "pause_duration" not in df.columns:
        df["pause_duration"] = np.nan
    if "phase" not in df.columns:
        df["phase"] = "train"  # traces written before the phase column existed

    return df.sort_values(["unique_episode", "steps"]).reset_index(drop=True)


def load_gym_traces(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate several runs, keeping ``unique_episode`` distinct.

    Each run keeps its own ``run`` label; ``unique_episode`` is offset per run
    so cross-run groupbys don't collide. Useful for comparing Optuna trials.
    """
    frames = []
    offset = 0
    for p in paths:
        d = load_gym_trace(p, episode_offset=offset)
        if len(d):
            frames.append(d)
            offset = int(d["unique_episode"].max()) + 1
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _infer_run_name(path: str) -> str:
    p = os.path.abspath(path)
    # walk up past trace/steps to the run dir name
    for marker in ("steps", "trace"):
        if os.path.basename(p) == marker:
            p = os.path.dirname(p)
    if p.endswith(".parquet"):
        p = os.path.dirname(os.path.dirname(os.path.dirname(p)))
    return os.path.basename(p.rstrip("/")) or "gym_run"


class GymTraceLog:
    """A thin, ``DeepRacerLog``-shaped wrapper around a dr-gym trace.

    Gives the gym trace the same ergonomics as a console/DRfC model folder so
    existing code paths work unchanged::

        from deepracer.logs import GymTraceLog
        log = GymTraceLog("artifacts/my_run").load()
        df = log.dataframe()
        stab = log.stability.analyze()        # SimtraceStabilityAnalyzer
    """

    def __init__(self, path: str):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self, force: bool = False) -> "GymTraceLog":
        if self.df is None or force:
            self.df = load_gym_trace(self.path)
        return self

    def dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise Exception("Trace not loaded; call load() first.")
        return self.df

    @property
    def stability(self):
        from .stability import SimtraceStabilityAnalyzer

        return SimtraceStabilityAnalyzer(self.dataframe())


# --------------------------------------------------------------------------- #
# Track-path plotting ("where did the car drive")
# --------------------------------------------------------------------------- #

def load_track(name: str, tracks_dir: Optional[str] = None):
    """Load a :class:`~deepracer.tracks.track_utils.Track` by name.

    Looks for ``<name>.npy`` (the (N, 6) center/inner/outer waypoint array) in,
    in order: *tracks_dir* arg, ``$DEEPRACER_TRACKS_DIR``, the track ``.npy``
    files shipped as package data (``deepracer/tracks/data`` — ``reinvent_base``
    out of the box), then the repo's test data (dev checkouts). Point
    *tracks_dir* at a directory of track ``.npy`` files (e.g. from the deepracer
    track-geometry repo) for other worlds.
    """
    import deepracer.tracks as _tracks_pkg
    from deepracer.tracks import TrackIO

    pkg_data = os.path.join(os.path.dirname(_tracks_pkg.__file__), "data")
    candidates = [
        tracks_dir,
        os.environ.get("DEEPRACER_TRACKS_DIR"),
        # Shipped as package data — present in a wheel install.
        pkg_data,
        # Repo test data — present only in a source checkout.
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "..", "tests", "deepracer", "track_utils", "tracks"),
    ]
    for base in candidates:
        if base and os.path.isfile(os.path.join(base, f"{name}.npy")):
            return TrackIO(base_path=base).load_track(name)
    raise FileNotFoundError(
        f"track {name!r}.npy not found. Pass tracks_dir=... or set "
        f"DEEPRACER_TRACKS_DIR to a folder of <track>.npy waypoint files."
    )


def _episode_frame(df: pd.DataFrame, episode: int) -> pd.DataFrame:
    col = "unique_episode" if "unique_episode" in df.columns else "episode"
    return df[df[col] == episode]


def plot_episode_path(df: pd.DataFrame, track, episode: Optional[int] = None,
                      value_field: str = "speed", title: Optional[str] = None,
                      ax=None):
    """Plot a single episode's path on the track outline, coloured by a field.

    Args:
        df: a loaded gym trace (or any frame with ``x``, ``y``, ``unique_episode``).
        track: a :class:`Track` (see :func:`load_track`).
        episode: which ``unique_episode`` to draw. Default: the longest episode.
        value_field: per-step column used for the colour (``speed``, ``reward``,
            ``progress`` …).
        title: optional plot title.
        ax: optional matplotlib Axes to draw into.
    """
    import matplotlib.pyplot as plt
    from .log_utils import PlottingUtils

    col = "unique_episode" if "unique_episode" in df.columns else "episode"
    if episode is None:
        episode = int(df.groupby(col)["steps"].max().idxmax())
    ep = _episode_frame(df, episode)
    if ep.empty:
        raise ValueError(f"no rows for episode {episode}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    PlottingUtils.print_border(ax, track, color="lightgrey")
    sc = ax.scatter(ep["x"], ep["y"], c=ep[value_field], cmap="viridis", s=12, zorder=3)
    ax.plot(ep["x"], ep["y"], lw=0.4, color="grey", alpha=0.5, zorder=2)
    status = ep["episode_status"].iloc[-1] if "episode_status" in ep else "?"
    prog = ep["progress"].max() if "progress" in ep else float("nan")
    ax.set_title(title or f"Episode {episode} path — {status}, progress {prog:.1f}% "
                          f"({len(ep)} steps)")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.colorbar(sc, ax=ax, label=value_field)
    return ax


def last_eval_episode(df: pd.DataFrame) -> Optional[int]:
    """The ``unique_episode`` of the most recent evaluation episode, or None.

    Uses the ``phase`` column (``"eval"`` rows). Returns None when the trace has
    no eval-tagged episodes (e.g. traces written before the phase column, or
    runs with evaluation disabled).
    """
    if "phase" not in df.columns:
        return None
    ev = df[df["phase"] == "eval"]
    if ev.empty:
        return None
    col = "unique_episode" if "unique_episode" in df.columns else "episode"
    return int(ev[col].max())


def plot_last_eval_path(df: pd.DataFrame, track, value_field: str = "speed", ax=None):
    """Plot the car's path on the **last evaluation episode**.

    Falls back to the highest-progress episode when the trace has no
    eval-tagged episodes, so it always produces a useful "where did it drive"
    view. Returns the Axes.
    """
    ep = last_eval_episode(df)
    if ep is None:
        col = "unique_episode" if "unique_episode" in df.columns else "episode"
        ep = int(df.groupby(col)["progress"].max().idxmax())
        note = " (no eval episodes — best-progress shown)"
    else:
        note = " (last eval)"
    ax = plot_episode_path(df, track, episode=ep, value_field=value_field, ax=ax)
    ax.set_title(ax.get_title() + note)
    return ax
