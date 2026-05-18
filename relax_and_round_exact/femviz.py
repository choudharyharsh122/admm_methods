# fem_api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Plot helpers (no FEniCS)
# ----------------------------

def _build_tri_points_and_indices(n: int):
    """
    Triangles ordered as: for I=0..n-1, for J=0..n-1: L then U
      L(I,J) = [(I,J),(I+1,J),(I+1,J+1)]
      U(I,J) = [(I,J),(I+1,J+1),(I,J+1)]
    """
    triangles = []
    for I in range(n):
        for J in range(n):
            #triangles.append([(I, J), (I + 1, J), (I + 1, J + 1)])   # L
            #triangles.append([(I, J), (I + 1, J + 1), (I, J + 1)])   # U
            triangles.append([(J, I), (J + 1, I), (J + 1, I + 1)])   # L
            triangles.append([(J, I), (J + 1, I + 1), (J, I + 1)])   # U

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
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    cbar = plt.colorbar(im, ax=ax)
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
    #cbar = plt.colorbar(im, ax=ax)
    return im


# ----------------------------
# Data container + API
# ----------------------------

@dataclass
class FemModel:
    alpha: float
    n: int
    seed: int

    metadata: Dict[str, Any]
    objectives: Dict[str, float]
    arrays: Dict[str, np.ndarray]
    runtime_total: float

    # ---- numeric aliases (DISCRETE DEFAULT, no "_disc") ----

    @property
    def objective(self) -> float:
        return float(self.objectives["disc_objective"])

    @property
    def objective_cont(self) -> float:
        return float(self.objectives["cont_objective"])

    @property
    def compliance(self) -> float:
        return float(self.objectives["disc_compliance"])

    @property
    def compliance_cont(self) -> float:
        return float(self.objectives["cont_compliance"])

    # raw TV stored in file
    @property
    def tv_raw(self) -> float:
        return float(self.objectives["disc_TV"])

    @property
    def tv_raw_cont(self) -> float:
        return float(self.objectives["cont_TV"])

    # your reporting convention: sqrt(n) * TV / alpha
    @property
    def tv(self) -> float:
        return float((self.n) * self.tv_raw / self.alpha)

    @property
    def tv_cont(self) -> float:
        return float((self.n) * self.tv_raw_cont / self.alpha)

    # ---- array aliases (DISCRETE DEFAULT, no "_disc") ----

    @property
    def control(self) -> np.ndarray:
        return self.arrays["a_disc"]

    @property
    def control_cont(self) -> np.ndarray:
        return self.arrays["a_opt"]

    @property
    def state(self) -> np.ndarray:
        return self.arrays["u_disc"]

    @property
    def state_cont(self) -> np.ndarray:
        return self.arrays["u_opt"]

    # ---- small summary ----
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": ["Objective", "Compliance", "Total Variation"],
            "Continuous": [self.objective_cont, self.compliance_cont, self.tv_cont],
            "Discrete":   [self.objective,      self.compliance,      self.tv],
        })

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
            a = _transpose_control_triangles(a, self.n)

        # Use the "plt.figure(); ax = plt.gca()" style (often plays nicest with axes_grid1)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.clear()

        title = "Continuous control (triangles)" if cont else "Discrete control (triangles)"
        im = _plot_control_tripcolor(ax, self.n, a, title=title, cmap=cmap)

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
            u = u.reshape((self.n + 1, self.n + 1)).T.ravel()

        # Use this style (works well with axes_grid1)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.clear()

        title = "Continuous state u (nodes)" if cont else "Discrete state u (nodes)"
        im = _plot_state_pcolormesh(ax, self.n, u, title=title, cmap=cmap)

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



# ----------------------------
# Factory function
# ----------------------------

def femModel(
    alpha: float,
    n: int,
    seed: int = 0,
    base_dir: str = "fem_model_tri",
    group_key: str = "summary",
) -> FemModel:
    """
    Usage:
        fm = femModel(alpha=1e-5, n=16)

    Loads:
      base_dir/{alpha}/{n}.h5, group {group_key}

    Returns FemModel with:
      fm.objective, fm.control, fm.solution (discrete defaults)
      fm.objective_cont, fm.control_cont, fm.solution_cont
    """
    h5_path = os.path.join(base_dir, f"{alpha}", f"{n}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        if group_key not in h5f:
            raise KeyError(f"Group '{group_key}' not found. Available: {list(h5f.keys())}")

        grp = h5f[group_key]

        metadata = {
            "dim": grp.attrs.get("dim"),
            "alpha": grp.attrs.get("alpha"),
            "V_frac": grp.attrs.get("V_frac"),
            "runtime_total": grp.attrs.get("runtime_total"),
            "runtime_solver": grp.attrs.get("runtime_solver"),
            "runtime_disc": grp.attrs.get("runtime_disc"),
        }

        objectives = {
            "cont_objective": float(grp["cont_objective"][()]),
            "cont_TV": float(grp["cont_TV"][()]),
            "cont_compliance": float(grp["cont_compliance"][()]),
            "disc_objective": float(grp["disc_objective"][()]),
            "disc_TV": float(grp["disc_TV"][()]),
            "disc_compliance": float(grp["disc_compliance"][()]),
        }

        arrays = {
            "a_opt":  np.array(grp["a_opt"], dtype=float),
            "a_disc": np.array(grp["a_disc"], dtype=float),
            "u_opt":  np.array(grp["u_opt"], dtype=float),
            "u_disc": np.array(grp["u_disc"], dtype=float),
        }

    runtime_total = metadata.get("runtime_total")

    return FemModel(
        alpha=alpha,
        n=n,
        seed=seed,
        metadata=metadata,
        objectives=objectives,
        arrays=arrays,
        runtime_total=float(runtime_total) if runtime_total is not None else float("nan"),
    )
