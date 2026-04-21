# API-style loader + lazy access for ADMM multi-seed HDF5 files.
#
# Usage:
#   from admm_api import ADMM
#   admm = ADMM(alpha=1e-6, dim=16, suffix="tri")   # suffix optional
#
#   # Median (default) summary
#   admm.objective, admm.tv, admm.compliance
#
#   # Median (default) final iterates
#   admm.control, admm.control_cont, admm.state
#
#   # Best summary + final iterates
#   admm.objective_best, admm.tv_best, admm.compliance_best
#   admm.control_best, admm.control_cont_best, admm.state_best
#
#
#   m = admm.median_seed   # int
#   b = admm.best_seed     # int
#   admm.trial(m).series.objective            # full objective_list
#   admm.trial(m).iters.state                 # full u_list (loaded lazily)
#   admm.trial(b).meta["rho_final"]
#
# Notes:
# - This expects the file structure as:
#     /seed_{i}/objective_list, tv_list, compliance_list, infeas_list, ...
#     /seed_{i}/iters/a_list, b_list, u_list, lambda_list, gradL_list
#     /seed_{i}/pair_metrics/*
#     /seed_{i}/triplet_metrics/*
#     /summary/*  and  /summary/best_iters/*, /summary/median_iters/*
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import re

import h5py
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
from dolfin import UnitSquareMesh, FunctionSpace, Function, plot as fenics_plot



SeedLike = Union[int, str]


# ---------------------------
# Utilities
# ---------------------------

_SEED_RE = re.compile(r"^seed_(\d+)$")


def _seed_int_to_name(seed: int) -> str:
    return f"seed_{int(seed)}"


def _seed_name_to_int(seed_name: str) -> int:
    m = _SEED_RE.match(seed_name)
    if not m:
        raise ValueError(f"Invalid seed group name: {seed_name!r} (expected 'seed_<int>')")
    return int(m.group(1))


def _decode_h5_scalar(x) -> Any:
    """
    Decode HDF5 scalars that might come out as bytes or 0-d arrays.
    """
    # h5py can return numpy scalars, bytes, or arrays
    if isinstance(x, (np.ndarray,)):
        if x.shape == ():
            x = x[()]
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def _read_last(ds) -> float:
    """
    Read last element of 1D dataset robustly.
    """
    # h5py supports negative indexing on datasets, but this keeps it explicit
    n = ds.shape[0]
    return float(ds[n - 1])

def _build_fenics_spaces(dim: int):
    """
    Same spaces as fenicsModel:
      DG0 for control (2*dim*dim dofs on UnitSquareMesh(dim,dim))
      CG1 for state   ((dim+1)^2 dofs)
    """
    mesh = UnitSquareMesh(dim, dim)
    Vc = FunctionSpace(mesh, "DG", 0)
    Vu = FunctionSpace(mesh, "CG", 1)
    return mesh, Vc, Vu


def _as_fenics_function(vec, V: FunctionSpace) -> Function:
    f = Function(V)
    f.vector().set_local(vec)
    f.vector().apply("insert")
    return f


def plot_control_field(control_vec, dim: int, *, ax=None, title: str = "Control", figsize=(5, 4), show: bool = True):
    mesh, Vc, _ = _build_fenics_spaces(dim)
    a = np.asarray(control_vec, dtype=float).ravel()

    expected = 2 * dim * dim
    if a.size != expected:
        raise ValueError(f"Control length mismatch: expected {expected} (=2*{dim}*{dim}), got {a.size}")

    f = _as_fenics_function(a, Vc)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    plt.sca(ax)
    ax.clear()
    m = fenics_plot(f, edgecolor="none")
    ax.margins(0)
    ax.autoscale_view()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(m, cax=cax, shrink=0.85)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=14)
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if show and created_fig:
        plt.show()

    return ax, m


def plot_state_field(state_vec, dim: int, *, ax=None, title: str = "State", figsize=(5, 4), show: bool = True) -> None:
    """
    Generic FEniCS state plotter (CG1).
    state_vec must be length (dim+1)^2.
    """
    mesh, _, Vu = _build_fenics_spaces(dim)
    u = np.asarray(state_vec, dtype=float).ravel()

    expected = (dim + 1) * (dim + 1)
    if u.size != expected:
        raise ValueError(f"State length mismatch: expected {expected} (=(dim+1)^2), got {u.size}")

    f = _as_fenics_function(u, Vu)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True

    plt.sca(ax)
    ax.clear()
    m = fenics_plot(f, edgecolor="none")
    ax.margins(0)
    ax.autoscale_view()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(m, cax=cax, shrink=0.85)
    cbar.set_ticks(np.linspace(u.min(), u.max(), 6))
    cbar.ax.tick_params(labelsize=14)
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if show and created_fig:
        plt.show()

    return ax, m


# ---------------------------
# Views for a seed group
# ---------------------------

@dataclass(frozen=True)
class _SeriesView:
    """
    Access to per-iteration scalar series arrays under seed_{i}/.
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            if key not in grp:
                raise KeyError(f"{self.seed_name}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def objective(self) -> np.ndarray:
        return self._read("objective_list")

    @property
    def tv(self) -> np.ndarray:
        return self._read("tv_list")

    @property
    def compliance(self) -> np.ndarray:
        return self._read("compliance_list")

    @property
    def infeas(self) -> np.ndarray:
        return self._read("infeas_list")

    @property
    def funnel(self) -> np.ndarray:
        return self._read("funnel_list")

    @property
    def runtime_sub1(self) -> np.ndarray:
        return self._read("runtime_sub1_list")

    @property
    def runtime_sub2(self) -> np.ndarray:
        return self._read("runtime_sub2_list")

    @property
    def h_tvs(self) -> np.ndarray:
        # if present
        return self._read("h_tvs")


@dataclass(frozen=True)
class _ItersView:
    """
    Access to iterate matrices under seed_{i}/iters/.
    Convention mapping:
      a_list -> control
      b_list -> control_cont
      u_list -> state
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/iters"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def control(self) -> np.ndarray:
        return self._read("a_list")

    @property
    def control_cont(self) -> np.ndarray:
        return self._read("b_list")

    @property
    def state(self) -> np.ndarray:
        return self._read("u_list")

    @property
    def lam(self) -> np.ndarray:
        return self._read("lambda_list")

    @property
    def gradL(self) -> np.ndarray:
        return self._read("gradL_list")

    # Convenience finals
    @property
    def control_final(self) -> np.ndarray:
        x = self.control
        return x[-1].copy()

    @property
    def control_cont_final(self) -> np.ndarray:
        x = self.control_cont
        return x[-1].copy()

    @property
    def state_final(self) -> np.ndarray:
        x = self.state
        return x[-1].copy()


@dataclass(frozen=True)
class _PairsView:
    """
    Access to seed_{i}/pair_metrics/*
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/pair_metrics"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def sub1_obj_pairs(self) -> np.ndarray:
        return self._read("sub1_obj_pairs")

    @property
    def compliance_pairs(self) -> np.ndarray:
        return self._read("compliance_pairs")

    @property
    def sub1_penalty_pairs(self) -> np.ndarray:
        return self._read("sub1_penalty_pairs")

    @property
    def sub2_obj_pairs(self) -> np.ndarray:
        return self._read("sub2_obj_pairs")

    @property
    def tv_pairs(self) -> np.ndarray:
        return self._read("tv_pairs")

    @property
    def sub2_penalty_pairs(self) -> np.ndarray:
        return self._read("sub2_penalty_pairs")


@dataclass(frozen=True)
class _TripletsView:
    """
    Access to seed_{i}/triplet_metrics/*
    """
    h5_path: str
    seed_name: str

    def _read(self, key: str) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as h5f:
            base = f"{self.seed_name}/triplet_metrics"
            if base not in h5f:
                raise KeyError(f"{base} group not found in file.")
            grp = h5f[base]
            if key not in grp:
                raise KeyError(f"{base}/{key} not found in file.")
            return np.array(grp[key][()], dtype=float)

    @property
    def aug_lagr_triplets(self) -> np.ndarray:
        return self._read("aug_lagr_triplets")


@dataclass(frozen=True)
class Trial:
    """
    A lightweight, lazy view into a single seed_{k} group.
    """
    h5_path: str
    seed: int
    
    """
    JSON tree description for one seed-level Trial view.
    """
    def describe_tree(self) -> Dict[str, Any]:
        return {
            "class": "Trial",
            "seed": self.seed,
            "seed_name": self.seed_name,
            "attributes": {
                "meta": "HDF5 attributes from /seed_k",
                "objective_final": self.objective_final,
                "infeas_final": self.infeas_final,
            },
            "children": {
                "series": {
                    "type": "_SeriesView",
                    "fields": [
                        "objective", "tv", "compliance", "infeas",
                        "funnel", "runtime_sub1", "runtime_sub2", "h_tvs"
                    ]
                },
                "iters": {
                    "type": "_ItersView",
                    "fields": [
                        "control", "control_cont", "state",
                        "lam", "gradL",
                        "control_final", "control_cont_final", "state_final"
                    ]
                },
                "pairs": {
                    "type": "_PairsView",
                    "fields": [
                        "sub1_obj_pairs", "compliance_pairs", "sub1_penalty_pairs",
                        "sub2_obj_pairs", "tv_pairs", "sub2_penalty_pairs"
                    ]
                },
                "triplets": {
                    "type": "_TripletsView",
                    "fields": ["aug_lagr_triplets"]
                }
            }
        }

    @property
    def seed_name(self) -> str:
        return _seed_int_to_name(self.seed)

    @property
    def meta(self) -> Dict[str, Any]:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return dict(grp.attrs.items())

    @property
    def series(self) -> _SeriesView:
        return _SeriesView(self.h5_path, self.seed_name)

    @property
    def iters(self) -> _ItersView:
        return _ItersView(self.h5_path, self.seed_name)

    @property
    def pairs(self) -> _PairsView:
        return _PairsView(self.h5_path, self.seed_name)

    @property
    def triplets(self) -> _TripletsView:
        return _TripletsView(self.h5_path, self.seed_name)

    # Convenience: finals without loading full iters unless asked
    @property
    def objective_final(self) -> float:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return _read_last(grp["objective_list"])

    @property
    def infeas_final(self) -> float:
        with h5py.File(self.h5_path, "r") as h5f:
            grp = h5f[self.seed_name]
            return _read_last(grp["infeas_list"])


# ---------------------------
# ADMM top-level API
# ---------------------------

class ADMM:
    """
    Summary-first API:
      - objective/tv/compliance/control/control_cont/state refer to MEDIAN summary
      - *_best fields refer to BEST summary
      - median_seed / best_seed are ints to drill down with trial(seed)
      - trial(seed) returns a lazy view into that seed group

    File lookup:
      base_dir/random_seeds_run_data_{suffix}/{alpha}/{dim}.h5

    If you already have a full path, pass h5_path directly.
    """

    def __init__(
        self,
        alpha: float,
        dim: int,
        *,
        suffix: str = "",
        base_dir: str = "run_data_admm_mergesplit",
        h5_path: Optional[str] = None,
    ):
        self.alpha = float(alpha)
        self.dim = int(dim)
        self.suffix = str(suffix)

        if h5_path is None:
            base = f"{base_dir}_{suffix}" if suffix else base_dir
            new_dir = os.path.join(base, str(alpha))
            h5_path = os.path.join(new_dir, f"{dim}.h5")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        self.h5_path = h5_path

        # ---- Eager-load ONLY summary essentials (small) ----
        self._load_summary()

    # -----------------
    # Public fields (summary-first)
    # -----------------

    # Scalars (median default)
    objective: float
    tv: float
    compliance: float

    # Final iterates (median default)
    control: np.ndarray
    control_cont: np.ndarray
    state: np.ndarray

    # Scalars (best)
    objective_best: float
    tv_best: float
    compliance_best: float

    # Final iterates (best)
    control_best: np.ndarray
    control_cont_best: np.ndarray
    state_best: np.ndarray

    # Seeds (int default)
    median_seed: int
    best_seed: int

    # Optional names
    median_seed_name: str
    best_seed_name: str

    # -----------------
    # Summary loading
    # -----------------

    def _load_summary(self) -> None:
        with h5py.File(self.h5_path, "r") as h5f:
            if "summary" not in h5f:
                raise KeyError("H5 file has no 'summary' group. Did you create it after runs?")
            s = h5f["summary"]

            # Seed names may be bytes; decode to str
            med_name = _decode_h5_scalar(s["median_seed_name"][()])
            best_name = _decode_h5_scalar(s["best_seed_name"][()])
            self.median_seed_name = str(med_name)
            self.best_seed_name = str(best_name)

            self.median_seed = _seed_name_to_int(self.median_seed_name)
            self.best_seed = _seed_name_to_int(self.best_seed_name)

            # Scalars
            self.objective = float(s["median_objective"][()])
            self.tv = float(s["median_tv"][()])
            self.compliance = float(s["median_compliance"][()])

            self.objective_best = float(s["best_objective"][()])
            self.tv_best = float(s["best_tv"][()])
            self.compliance_best = float(s["best_compliance"][()])

            # Final iterates from summary groups
            self.control = np.array(s["median_iters"]["a_list"][-1], dtype=float)
            self.control_cont = np.array(s["median_iters"]["b_list"][-1], dtype=float)
            self.state = np.array(s["median_iters"]["u_list"][-1], dtype=float)

            self.control_best = np.array(s["best_iters"]["a_list"][-1], dtype=float)
            self.control_cont_best = np.array(s["best_iters"]["b_list"][-1], dtype=float)
            self.state_best = np.array(s["best_iters"]["u_list"][-1], dtype=float)

            # Keep a small metadata dict available (cheap)
            # Prefer summary attrs if present; otherwise leave empty.
            self.metadata = dict(s.attrs.items()) if hasattr(s, "attrs") else {}

    # -----------------
    # Discoverability
    # -----------------

    def describe(self) -> Dict[str, List[str]]:
        """
        Compact listing of core attributes + methods.
        """
        attrs = [
            "alpha", "dim", "suffix", "h5_path",
            "median_seed", "median_seed_name",
            "best_seed", "best_seed_name",
            "objective", "tv", "compliance",
            "control", "control_cont", "state",
            "objective_best", "tv_best", "compliance_best",
            "control_best", "control_cont_best", "state_best",
            "metadata",
        ]
        methods = [
            "trial(k) -> Trial",
            "seeds() -> List[int]",
            "trials_df() -> pd.DataFrame",
            "reload_summary()",
            "describe()",
        ]
        return {"attributes": attrs, "methods": methods}

    # -----------------
    # Trial access
    # -----------------

    def trial(self, seed: SeedLike) -> Trial:
        """
        Return a lazy view into seed_{k}.
        seed can be int (preferred) or string like 'seed_7'.
        """
        if isinstance(seed, str):
            seed = _seed_name_to_int(seed)
        seed = int(seed)

        # Validate existence quickly without loading data
        with h5py.File(self.h5_path, "r") as h5f:
            name = _seed_int_to_name(seed)
            if name not in h5f:
                raise KeyError(f"Seed group '{name}' not found. Available: {list(h5f.keys())}")
        return Trial(self.h5_path, seed)

    # -----------------
    # Seed inventory + lightweight summaries
    # -----------------

    def seeds(self) -> List[int]:
        """
        List available seed integers in the file.
        """
        with h5py.File(self.h5_path, "r") as h5f:
            out = []
            for k in h5f.keys():
                if k.startswith("seed_") and _SEED_RE.match(k):
                    out.append(_seed_name_to_int(k))
            return sorted(out)

    def trials_df(self, infeas_thresh: Optional[float] = None) -> pd.DataFrame:
        """
        Build a lightweight per-seed table of final metrics.
        Does NOT load iterates.
        If infeas_thresh is provided, include a boolean column 'feasible'.
        """
        rows = []
        with h5py.File(self.h5_path, "r") as h5f:
            for k in h5f.keys():
                if not (k.startswith("seed_") and _SEED_RE.match(k)):
                    continue
                grp = h5f[k]
                seed = _seed_name_to_int(k)

                obj_last = _read_last(grp["objective_list"])
                tv_last = _read_last(grp["tv_list"])
                comp_last = _read_last(grp["compliance_list"])
                infeas_last = _read_last(grp["infeas_list"])

                row = {
                    "seed": seed,
                    "objective_last": obj_last,
                    "tv_last": tv_last,
                    "compliance_last": comp_last,
                    "infeas_last": infeas_last,
                }

                # common attrs if present
                for attr_key in ["final_iter", "rho_init", "rho_final", "zeta", "beta", "idx", "backend", "dim", "alpha"]:
                    if attr_key in grp.attrs:
                        row[attr_key] = grp.attrs[attr_key]

                rows.append(row)

        df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
        if infeas_thresh is not None:
            df["feasible"] = df["infeas_last"] <= float(infeas_thresh)
        return df

    # -----------------
    # Refresh
    # -----------------

    def reload_summary(self) -> None:
        """
        Re-read summary from disk (useful if you update the H5 file).
        """
        self._load_summary()

    
    def plot_control(
        self,
        control: Optional[np.ndarray] = None,
        *,
        cont: bool = False,
        best: bool = False,
        ax=None,
        figsize=(6, 5),
        title: Optional[str] = None,
        show: bool = True,
        ):
        """
        Plot a DG0 control as a FEniCS Function.

        Behavior:
          - If `control` is provided: plot that array (ignores best/cont selection except for default title).
          - Else:
              best=False -> plot median final control (a) or control_cont (b) if cont=True
              best=True  -> plot best   final control (a) or control_cont (b) if cont=True
        """
        if control is not None:
            vec = np.asarray(control, dtype=float).ravel()
            if title is None:
                title = f"Control (provided, {'cont' if cont else 'disc'})"
            plot_control_field(vec, dim=self.dim, title=title, figsize=figsize, ax=ax, show=show)
            return

        # choose from stored finals
        if best:
            vec = self.control_cont_best if cont else self.control_best
            which = "Best"
        else:
            vec = self.control_cont if cont else self.control
            which = "Median"

        if title is None:
            title = f"{which} control ({'cont' if cont else 'disc'})"

        plot_control_field(vec, dim=self.dim, title=title, figsize=figsize, ax=ax, show=show)

    def plot_state(
        self,
        state: Optional[np.ndarray] = None,
        *,
        cont: bool = False,
        best: bool = False,
        ax=None,
        figsize=(6, 5),
        title: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot a CG1 state as a FEniCS Function.

        Behavior:
          - If `state` is provided: plot that array (ignores best/cont selection except for default title).
          - Else:
              best=False -> plot median final state
              best=True  -> plot best   final state

        Note:
          `cont` is accepted for API symmetry but has no effect (no state_cont exists).
        """
        if state is not None:
            vec = np.asarray(state, dtype=float).ravel()
            if title is None:
                title = "State (provided)"
            plot_state_field(vec, dim=self.dim, ax=ax, title=title, figsize=figsize, show=show)
            return

        # choose from stored finals
        if best:
            vec = self.state_best
            which = "Best"
        else:
            vec = self.state
            which = "Median"

        if title is None:
            title = f"{which} state"

        # cont intentionally ignored
        plot_state_field(vec, dim=self.dim, ax=ax, title=title, figsize=figsize, show=show)

