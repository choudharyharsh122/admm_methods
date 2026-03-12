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


def _build_tri_points_and_indices(n: int):
    """
    Triangles ordered as: for I=0..n-1, for J=0..n-1: L then U
      L(I,J) = [(I,J),(I+1,J),(I+1,J+1)]
      U(I,J) = [(I,J),(I+1,J+1),(I,J+1)]
    """
    triangles = []
    for I in range(n):
        for J in range(n):
            triangles.append([(I, J), (I + 1, J), (I + 1, J + 1)])   # L
            triangles.append([(I, J), (I + 1, J + 1), (I, J + 1)])   # U

    points = []
    tri_indices = []
    for tri in triangles:
        idxs = []
        for (i, j) in tri:
            points.append([i, j])
            idxs.append(len(points) - 1)
        tri_indices.append(idxs)

    return np.array(points, float), np.array(tri_indices, int)

def _transpose_control_triangles(a: np.ndarray, n: int) -> np.ndarray:
    """
    Fix global diagonal reflection (transpose) for triangle-wise controls.
    Maps (I,J) -> (J,I) for both L and U entries.
    """
    a = np.asarray(a, dtype=float)
    if a.size != 2 * n * n:
        raise ValueError(f"Expected control length {2*n*n}, got {a.size}")

    out = np.empty_like(a)
    for I in range(n):
        for J in range(n):
            src = 2 * (J + I * n)      # swapped
            dst = 2 * (I + J * n)
            out[dst] = a[src]          # L
            out[dst + 1] = a[src + 1]  # U
    return out


def _plot_control_tripcolor(ax, n: int, a: np.ndarray, title: str, cmap: str = "viridis"):
    points, tri_indices = _build_tri_points_and_indices(n)
    im = ax.tripcolor(points[:, 0], points[:, 1], tri_indices, a,
                      shading="flat", cmap=cmap)
    #ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    return im


def _plot_state_pcolormesh(ax, n: int, u: np.ndarray, title: str, cmap: str = "viridis"):
    u = np.asarray(u, dtype=float)
    expected = (n + 1) * (n + 1)
    if u.size != expected:
        raise ValueError(f"Expected state length {expected}, got {u.size}")

    X, Y = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    u_grid = u.reshape((n + 1, n + 1))

    im = ax.pcolormesh(X, Y, u_grid, shading="gouraud", cmap=cmap)
    #ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    return im


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


# def plot_control(control_vec, dim: int, *, title: str = "Control", figsize=(5, 4)) -> None:
#     """
#     Generic FEniCS control plotter (DG0).
#     control_vec must be length 2*dim*dim.
#     """
#     mesh, Vc, _ = _build_fenics_spaces(dim)
#     a = np.asarray(control_vec, dtype=float).ravel()

#     expected = 2 * dim * dim
#     if a.size != expected:
#         raise ValueError(f"Control length mismatch: expected {expected} (=2*{dim}*{dim}), got {a.size}")

#     f = _as_fenics_function(a, Vc)

#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plt.sca(ax)
#     ax.clear()
#     m = fenics_plot(f, edgecolor='none')

#     #ax.set_aspect("equal", adjustable="box")
#     ax.margins(0)              # remove internal padding
#     ax.autoscale_view()        # rescale to data

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="4%", pad=0.1)

#     cbar = plt.colorbar(m, cax=cax, shrink=0.85)
#     #cbar = plt.colorbar(m, ax=ax)
#     cbar.set_ticks(np.linspace(0, 1, 6))
#     cbar.ax.tick_params(labelsize=14)
#     #ax.set_title(title)
#     ax.set_axis_off()
#     #plt.tight_layout()
    
#     plt.show()


# def plot_state(state_vec, dim: int, *, title: str = "State", figsize=(5, 4)) -> None:
#     """
#     Generic FEniCS state plotter (CG1).
#     state_vec must be length (dim+1)^2.
#     """
#     mesh, _, Vu = _build_fenics_spaces(dim)
#     u = np.asarray(state_vec, dtype=float).ravel()

#     expected = (dim + 1) * (dim + 1)
#     if u.size != expected:
#         raise ValueError(f"State length mismatch: expected {expected} (=(dim+1)^2), got {u.size}")

#     f = _as_fenics_function(u, Vu)

#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     plt.sca(ax)
#     ax.clear()
#     m = fenics_plot(f)
#     ax.set_aspect("equal", adjustable="box")
#     ax.margins(0)              # remove internal padding
#     ax.autoscale_view()        # rescale to data
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="4%", pad=0.1)

#     cbar = plt.colorbar(m, cax=cax, shrink=0.85)
#     #cbar = plt.colorbar(m, ax=ax)
#     cbar.ax.tick_params(labelsize=14)
#     #ax.set_title(title)
#     ax.set_axis_off()
#     #ax.set_title(title)
#     #plt.tight_layout()
#     plt.show()


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

    # @property
    # def gradL(self) -> np.ndarray:
    #     return self._read("gradL_list")

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

    
    # ---- plotting API (discrete by default) ----
    def plot_control(
        self,
        cont: bool = False,
        fix_diagonal_reflection: bool = True,
        cmap: str = "viridis",
        figsize=(6, 5),
    ) -> None:
        """
        Plot control as triangle-wise tripcolor.
        Discrete by default unless cont=True.
        Match the "fenics_plot" styling: no axis box, tight data limits,
        square aspect, and a colorbar whose height matches the axes.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        a = self.control_cont if cont else self.control
        a = np.asarray(a, dtype=float)

        if fix_diagonal_reflection:
            a = _transpose_control_triangles(a, self.dim)

        # Use the "plt.figure(); ax = plt.gca()" style (often plays nicest with axes_grid1)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.clear()

        title = "Continuous control (triangles)" if cont else "Discrete control (triangles)"
        im = _plot_control_tripcolor(ax, self.dim, a, title=title, cmap=cmap)

        # --- Make the plotted field fill the axes (tight limits, no padding) ---
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0)

        # For tripcolor/collections, autoscale can be finicky; do both:
        ax.relim()
        ax.autoscale_view(tight=True)

        # If your _plot_control_tripcolor sets limits with padding, this overrides it.
        # (Keep this; it helps a lot for "plot inside axis" whitespace.)
        try:
            ax.set_xlim(*im.axes.dataLim.intervalx)
            ax.set_ylim(*im.axes.dataLim.intervaly)
        except Exception:
            pass

        # --- Colorbar matched to axes height + tight gap ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=14)
        #cbar.set_label("w", fontsize=14)

        # Remove the square border/ticks entirely
        ax.set_axis_off()

        # Avoid tight_layout here (it can reintroduce padding with appended axes)
        plt.show()


    def plot_state(
    self,
    cont: bool = False,
    transpose: bool = False,
    cmap: str = "viridis",
    figsize=(6, 5),
) -> None:
        """
        Plot state as node-wise pcolormesh.
        Discrete by default unless cont=True.
        If transpose=True, plot u reshaped then transposed.

        Styled to match your compact FEniCS plots:
        - square aspect
        - no axis box/ticks
        - plot fills the axes (no internal padding)
        - colorbar height matches axes and sits close
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        u = self.state_cont if cont else self.state
        u = np.asarray(u, dtype=float)

        if transpose:
            u = u.reshape((self.dim + 1, self.dim + 1)).T.ravel()

        # Use this style (works well with axes_grid1)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.clear()

        title = "Continuous state u (nodes)" if cont else "Discrete state u (nodes)"
        im = _plot_state_pcolormesh(ax, self.dim, u, title=title, cmap=cmap)

        # --- Make the plotted field fill the axes tightly ---
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0)

        # For pcolormesh, this reliably tightens limits to the mesh extent
        ax.relim()
        ax.autoscale_view(tight=True)

        # Remove any extra padding that some helper functions introduce
        try:
            ax.set_xlim(*im.axes.dataLim.intervalx)
            ax.set_ylim(*im.axes.dataLim.intervaly)
        except Exception:
            pass

        # --- Colorbar: same height as the axes, small gap ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=14)
        #cbar.set_label("u(x)", fontsize=14)

        # No border/ticks
        ax.set_axis_off()

        # Avoid tight_layout (it can fight the appended cax)
        plt.show()

