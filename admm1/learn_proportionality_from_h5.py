import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


def _as_float(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            text = text.split(",")[0].strip()
        return float(text)
    return float(value)


def load_h5_records(run_root: Path, dim_filter=None):
    records = []
    for h5_path in sorted(run_root.rglob("*.h5")):
        with h5py.File(h5_path, "r") as handle:
            for name, group in handle.items():
                if not name.startswith("seed_"):
                    continue

                dim = int(group.attrs.get("dim", -1))
                if dim_filter is not None and dim != dim_filter:
                    continue

                alpha = _as_float(group.attrs["alpha"])
                rho = _as_float(group.attrs.get("rho_final", group.attrs.get("rho_init")))
                objective = float(group["objective_list"][-1])
                infeasibility = float(group["infeas_list"][-1])

                records.append(
                    {
                        "alpha": alpha,
                        "rho": rho,
                        "objective": objective,
                        "infeasibility": infeasibility,
                        "dim": dim,
                    }
                )
    return records


def fit_linear_xy(x: np.ndarray, y: np.ndarray):
    # y = b0 + b1*x1 + b2*x2
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def predict_linear_xy(x: np.ndarray, coef: np.ndarray):
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coef


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_alpha_equals_c_rho(alpha: np.ndarray, rho: np.ndarray):
    # alpha = c * rho (through origin)
    denom = float(np.dot(rho, rho))
    c = float(np.dot(rho, alpha) / max(denom, 1e-12))
    return c


def main():
    parser = argparse.ArgumentParser(
        description="Simple linear models from HDF5 and fit alpha = c * rho."
    )
    parser.add_argument("--run-root", type=Path, default=Path("run_data_admm_gurobi"))
    parser.add_argument("--dim", type=int, default=20, help="Use -1 for all dimensions")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default=Path("learned_alpha_rho_models.pkl"))
    args = parser.parse_args()

    dim_filter = None if args.dim < 0 else args.dim
    records = load_h5_records(args.run_root, dim_filter=dim_filter)
    if len(records) < 5:
        raise SystemExit(f"Not enough records found. Found: {len(records)}")

    alpha = np.array([r["alpha"] for r in records], dtype=float)
    rho = np.array([r["rho"] for r in records], dtype=float)
    y_obj = np.array([r["objective"] for r in records], dtype=float)
    y_inf = np.array([r["infeasibility"] for r in records], dtype=float)

    x = np.column_stack([alpha, rho])

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)

    split = int(max(1, min(len(records) - 1, round(args.train_frac * len(records)))))
    tr = idx[:split]
    va = idx[split:]

    x_tr, x_va = x[tr], x[va]
    y_obj_tr, y_obj_va = y_obj[tr], y_obj[va]
    y_inf_tr, y_inf_va = y_inf[tr], y_inf[va]

    obj_coef = fit_linear_xy(x_tr, y_obj_tr)
    inf_coef = fit_linear_xy(x_tr, y_inf_tr)

    obj_rmse_tr = rmse(y_obj_tr, predict_linear_xy(x_tr, obj_coef))
    inf_rmse_tr = rmse(y_inf_tr, predict_linear_xy(x_tr, inf_coef))

    obj_rmse_va = rmse(y_obj_va, predict_linear_xy(x_va, obj_coef)) if len(va) else float("nan")
    inf_rmse_va = rmse(y_inf_va, predict_linear_xy(x_va, inf_coef)) if len(va) else float("nan")

    c_train = fit_alpha_equals_c_rho(alpha[tr], rho[tr])
    c_all = fit_alpha_equals_c_rho(alpha, rho)

    print(f"Loaded records: {len(records)}")
    print(f"Train/val split: {len(tr)}/{len(va)}")
    print(f"Objective model RMSE (train/val): {obj_rmse_tr:.6e} / {obj_rmse_va:.6e}")
    print(f"Infeasibility model RMSE (train/val): {inf_rmse_tr:.6e} / {inf_rmse_va:.6e}")
    print(f"Learned proportionality (train): alpha ~= {c_train:.6f} * rho")
    print(f"Learned proportionality (all):   alpha ~= {c_all:.6f} * rho")

    payload = {
        "records": records,
        "objective_model": {
            "form": "objective = b0 + b1*alpha + b2*rho",
            "coef": obj_coef,
            "rmse_train": obj_rmse_tr,
            "rmse_val": obj_rmse_va,
        },
        "infeasibility_model": {
            "form": "infeasibility = b0 + b1*alpha + b2*rho",
            "coef": inf_coef,
            "rmse_train": inf_rmse_tr,
            "rmse_val": inf_rmse_va,
        },
        "proportionality": {
            "form": "alpha = c * rho",
            "c_train": c_train,
            "c_all": c_all,
        },
        "split": {
            "train_frac": args.train_frac,
            "seed": args.seed,
            "train_count": int(len(tr)),
            "val_count": int(len(va)),
        },
    }

    with open(args.out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved to: {args.out_file}")


if __name__ == "__main__":
    main()
