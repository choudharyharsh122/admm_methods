import argparse
import math
import pickle
import time
from typing import Dict, Tuple, List
import os
import h5py
import random


import numpy as np
import networkx as nx  
from cyipopt import Problem  

# Local imports
from design_variables import DesignVariables

# To track convergence of OC, F and \nabla F
TRACK_OC_CONVERGENCE = True
from subproblem1_solver import *
from subproblem2_solver import Subproblem2Solver


def find_best_and_median_seeds(h5f, infeas_thresh):
    """
    Scan all seed_* groups in h5 file and find:
      - best seed (min final objective)
      - median seed (median final objective)
    Only seeds with final infeasibility <= infeas_thresh are considered.

    Fallback:
      - If no seed satisfies infeas_thresh, return the seed with
        minimum final infeasibility (for both best and median).
    """
    seed_names = [k for k in h5f.keys() if k.startswith("seed_")]

    objs = []
    infeas = []
    names = []

    for name in seed_names:
        grp = h5f[name]
        objs.append(grp["objective_list"][-1])
        infeas.append(grp["infeas_list"][-1])
        names.append(name)

    objs = np.asarray(objs, float)
    infeas = np.asarray(infeas, float)

    # seeds satisfying infeasibility threshold
    valid = infeas <= infeas_thresh

    if not np.any(valid):
        # --- fallback: minimum infeasibility ---
        best_pos = np.argmin(infeas)
        best_seed = names[best_pos]
        median_seed = best_seed
        return best_seed, median_seed

    # restrict to valid seeds
    valid_objs = objs[valid]
    valid_names = [n for n, v in zip(names, valid) if v]

    # best among valid
    best_pos = np.argmin(valid_objs)
    best_seed = valid_names[best_pos]

    # median among valid
    order = np.argsort(valid_objs)
    median_pos = order[len(order) // 2]
    median_seed = valid_names[median_pos]

    return best_seed, median_seed




def run_trial(dim: int, idx: int, params) -> None:
    """
    Runs one mesh-size trial and writes results into an HDF5 file (key-value structure).
    """
    print(f"=== TRIAL {idx} | dim={dim} ===", flush=True)

    
    V_max = params.vol_frac
    f = params.source_strength
    
    # Determine number of seeds based on backend
    backend = (params.backend or "").lower()
    if backend == "mergesplit":
        num_seeds = 10
        print(f"Using mergesplit backend: running {num_seeds} random seeds")
    elif backend == "gurobi":
        num_seeds = 1
        print(f"Using gurobi backend: running {num_seeds} seed (deterministic solver)")
    else:
        num_seeds = 10
        print(f"Unknown backend '{backend}': defaulting to {num_seeds} seeds")
    
    seeds = [random.randint(0, 10000) for _ in range(num_seeds)]

    alpha_list = [float(x) for x in params.alpha.split(",")]
    rho_list = [float(x) for x in params.rho.split(",")]

    for alpha,rho in zip(alpha_list, rho_list):
        print(f"\n--- Running with alpha={alpha} ---")
        
        for (i, seed_i) in enumerate(seeds):

            # --- ADMM variables
            dv = DesignVariables(seed=42, size=dim, Vmax=V_max)
            a_k, b_k, lam_k = dv.a, dv.b, dv.lam
            rho_k = rho

            print(f"======= Running for seed number {i}, value {seed_i}")

            # --- Solver setup
            #core = Solver(mesh, A, P, bc, f, alpha, V_max)
            sub2 = Subproblem2Solver(n_x=dim, n_y=2 * dim, alpha=alpha, seed=seed_i)
            sub1 = Subproblem1Solver(dim, f, volfrac=params.vol_frac, alpha=alpha, graph=sub2.graph, scale=sub2.scale)

            # --- Funnel initialization
            eta_k = np.linalg.norm(a_k - b_k) ** 2
            tau_k = max(1.0, eta_k) * params.gamma

            # --- Data storage lists, regular list storing objectives at iter k
            a_list, b_list, u_list, lam_list, grad_list = [], [], [], [], []
            obj_list, tv_list, compliance_list, infeas_list = [], [], [], []
            funnel_list = []

            runtime1_list, runtime2_list = [], []

            # OC convergence tracking for each subproblem-1 solve call
            # (each item corresponds to one accepted ADMM iterate).
            oc_track_per_admm_iter = []

            # infeas_k = (1 / len(b_k)) * (np.linalg.norm(a_k - b_k) ** 2)
            # print(f"> Infeasibility (init): {infeas_k:.6e}")
            # tol = min(1e-3, infeas_k / 5)
            # print(f">> tolerance for the run is : {tol:.6e}")

            # ---- Initial values of iterates ---
            a_list.append(a_k)
            b_list.append(b_k)
            lam_list.append(lam_k)
            infeas_list.append(np.linalg.norm(a_k - b_k)**2)
            funnel_list.append(params.beta * tau_k)
            #grad_list.append(0.4*np.ones(len(b_k)))
            u_list.append(np.zeros((dim+1)*(dim+1)))
            sub1_obj_k, compliance_k, pen_k, gradL_k = sub1.compute_Objs(a_k, b_k, lam_k, rho_k)
            compliance_list.append(compliance_k)
            tv_list.append(sub2.compute_TV(a_k, b_k, lam_k, rho_k))
            obj_list.append(compliance_list[-1] + tv_list[-1])

            l = 0

            while l < params.break_iter:
                # --- Subproblem 1 ---
                t0 = time.perf_counter()
                b_k1, u_k1, oc_track_sub1 = sub1.solve(a_k, b_k, lam_k, rho_k, track_oc_convergence=TRACK_OC_CONVERGENCE)
                runtime1_list.append(time.perf_counter() - t0)

                # --- Subproblem 2 ---
                t0 = time.perf_counter()
                # --- A loop to try with small perturbations if integer solver doesn't find a solution
                while True:
                    sol, status = sub2.run(a_k, b_k1, lam_k, rho_k, V_max, seed_i, params.backend)
                    if sol is None:
                        print(">>> solver failed; retry with modified a_k (sparsify) <<<")
                        eligible = np.where(b_k1 < 0.5)[0]
                        if eligible.size > 0:
                            pick = np.random.choice(eligible, size=max(1, int(0.05 * eligible.size)), replace=False)
                            a_k[pick] = 0
                        else:
                            break
                    else:
                        a_k1 = sol.copy()
                        break
                runtime2_list.append(time.perf_counter() - t0)

                
                eta_k = np.linalg.norm(b_k1 - a_k1) ** 2
                
                # --- Current iterate accepted in the funnel
                if eta_k <= params.beta * tau_k:
                    
                    print(f"Step {l} accepted.")
                    
                    # --- dual update step (regular ADMM) ---
                    lam_k1 = lam_k + rho_k * (b_k1 - a_k1)
                    # --- reduce funnel ---
                    tau_k = (1 - params.zeta) * eta_k + params.zeta * tau_k  # EMA update

                    '''----------Compute everything----------
                    This includes computing Compliance, TV and Objective values
                    '''
                    sub1_obj_k1, compliance_k1, pen_k1, gradL_k1 = sub1.compute_Objs(a_k1, a_k1, lam_k1, rho_k)
                    u_k1 = sub1._solve_state(a_k1)
                    tv_k1 = sub2.compute_TV(a_k1, b_k1, lam_k1, rho_k)


                    # store iterates
                    a_list.append(a_k1.copy())
                    b_list.append(b_k1.copy())
                    lam_list.append(lam_k1.copy())
                    # gradient is the value \nabla_b L_{\rho} (with respect to continuous variable)
                    
                    u_list.append(u_k1.copy())
                    funnel_list.append(params.beta * tau_k)
                    infeas_list.append(np.linalg.norm(b_k1 - a_k1)**2)
                     
                    
                    # --- Store regular list at iter k
                    # --- Thee are our objectives that we actually care about
                    compliance_list.append(compliance_k1)
                    #print(">>>>> TV is:", (np.sqrt(len(a_k1)/2))*tv_k1/alpha)
                    tv_list.append((np.sqrt(len(a_k1)/2))*tv_k1/alpha)
                    obj_list.append(compliance_k1 + tv_k1)
                    
                     
                    if TRACK_OC_CONVERGENCE and oc_track_sub1 is not None:
                        oc_track_per_admm_iter.append(oc_track_sub1)

                    ''' Update iterates'''
                    a_k, b_k, lam_k = a_k1, b_k1, lam_k1

                else:
                    # --- increase penalty (regular ADMM: lambda is not rescaled)
                    rho_k *= 1.25
                    print(f"Step {l} rejected ? increasing penalty (rho={rho_k:.3e}).", flush=True)

                l += 1
                infeas_k = (np.linalg.norm(a_k - b_k) ** 2)
                print(f"> Infeasibility: {infeas_k:.6e}", flush=True)
                
            # --- Save
            base_dir = f"run_data_admm_{backend}"
            new_dir = os.path.join(base_dir, str(alpha))
            os.makedirs(new_dir, exist_ok=True)
            h5_path = os.path.join(new_dir, f"{dim}.h5")

            print(f"Saving results to {h5_path}")

            with h5py.File(h5_path, "a") as h5f:  # 'a' = append mode
                # Create top-level group for this seed
                seed_group = h5f.create_group(f"seed_{i}")

                # --- Metadata ---
                seed_group.attrs["dim"] = dim
                seed_group.attrs["idx"] = idx
                seed_group.attrs["backend"] = backend
                seed_group.attrs["final_iter"] = l
                seed_group.attrs["alpha"] = alpha
                seed_group.attrs["rho_init"] = params.rho
                seed_group.attrs["rho_final"] = rho_k
                seed_group.attrs["zeta"] = params.zeta
                seed_group.attrs["beta"] = params.beta

                # --- Scalar & simple lists ---
                seed_group.create_dataset("objective_list", data=np.array(obj_list, dtype=np.float64))
                seed_group.create_dataset("tv_list", data=np.array(tv_list, dtype=np.float64))
                seed_group.create_dataset("compliance_list", data=np.array(compliance_list, dtype=np.float64))
                seed_group.create_dataset("runtime_sub1_list", data=np.array(runtime1_list, dtype=np.float64))
                seed_group.create_dataset("runtime_sub2_list", data=np.array(runtime2_list, dtype=np.float64))
                seed_group.create_dataset("funnel_list", data=np.array(funnel_list, dtype=np.float64))
                seed_group.create_dataset("infeas_list", data=np.array(infeas_list, dtype=np.float64))

                # --- Iterates ---
                grp_iters = seed_group.create_group("iters")
                grp_iters.create_dataset("a_list", data=np.vstack(a_list))
                grp_iters.create_dataset("b_list", data=np.vstack(b_list))
                grp_iters.create_dataset("u_list", data=np.vstack(u_list))
                grp_iters.create_dataset("lambda_list", data=np.vstack(lam_list))
                #grp_iters.create_dataset("gradL_list", data=np.vstack(grad_list))

                # OC convergence data (saved only when flag is enabled)
                if TRACK_OC_CONVERGENCE and len(oc_track_per_admm_iter) > 0:
                    grp_oc = seed_group.create_group("oc_subproblem1_tracking")
                    for admm_iter_idx, track_data in enumerate(oc_track_per_admm_iter):
                        grp_iter = grp_oc.create_group(f"admm_iter_{admm_iter_idx}")
                        grp_iter.create_dataset("F_list", data=np.asarray(track_data["F_list"], dtype=np.float64))
                        grp_iter.create_dataset("grad_F_list", data=np.asarray(track_data["grad_F_list"], dtype=np.float64))
                        grp_iter.create_dataset("grad_F_norm_list", data=np.asarray(track_data["grad_F_norm_list"], dtype=np.float64))
            print(f">> Added results for seed={seed_i} (dim={dim}, {l} iterations)")
        
        # --- Save summary data ---
        # Summary contains the best and median performance seeds out of the randomly selected samples
        with h5py.File(h5_path, "a") as h5f:
            summary = h5f.require_group("summary")

            def save(name, data):
                if name in summary:
                    del summary[name]
                summary.create_dataset(name, data=data)

            # For gurobi (single seed), use that seed for both best and median
            # Deterministic case
            if num_seeds == 1:
                # Single seed case (gurobi)
                seed_name = "seed_0"
                grp = h5f[seed_name]
                
                # Save as both best and median (since there's only one)
                save("best_seed_name", np.bytes_(seed_name))
                save("best_objective", grp["objective_list"][-1])
                save("best_tv", grp["tv_list"][-1])
                save("best_compliance", grp["compliance_list"][-1])
                
                save("median_seed_name", np.bytes_(seed_name))
                save("median_objective", grp["objective_list"][-1])
                save("median_tv", grp["tv_list"][-1])
                save("median_compliance", grp["compliance_list"][-1])
                
                # Save iterates for both best and median (same data)
                best_iters = summary.require_group("best_iters")
                for k in list(best_iters.keys()):
                    del best_iters[k]
                for name in grp["iters"]:
                    best_iters.create_dataset(name, data=grp["iters"][name][()])
                
                med_iters = summary.require_group("median_iters")
                for k in list(med_iters.keys()):
                    del med_iters[k]
                for name in grp["iters"]:
                    med_iters.create_dataset(name, data=grp["iters"][name][()])
                    
                print(f"\n=== Summary saved (single seed: {seed_name}) ===")

            # For mergesplit (multiple seeds), find best and median
            # Randomized case    
            else:
                # Multiple seeds case (mergesplit)
                # The second argument is infeasibility tolerance for accepting a solution as a good candidate
                # if \|v-w\|^2 <= \epsilon, then we use this seed as a representative of the entire sample space
                # The ADMM algorithm provably converges in infeasibility i.e. \epsilon goes to 0
                # for all possible \rho_0 and is independent of dim but since we are only running
                # the trials for a fixed number of iterations, 
                # the problem size, we use this as a safe choice. 
                best_seed, median_seed = find_best_and_median_seeds(h5f, (dim)/2)

                # --- Best ---
                grp_best = h5f[best_seed]
                save("best_seed_name", np.bytes_(best_seed))
                save("best_objective", grp_best["objective_list"][-1])
                save("best_tv", grp_best["tv_list"][-1])
                save("best_compliance", grp_best["compliance_list"][-1])

                best_iters = summary.require_group("best_iters")
                for k in list(best_iters.keys()):
                    del best_iters[k]
                for name in grp_best["iters"]:
                    best_iters.create_dataset(name, data=grp_best["iters"][name][()])

                # --- Median ---
                grp_med = h5f[median_seed]
                save("median_seed_name", np.bytes_(median_seed))
                save("median_objective", grp_med["objective_list"][-1])
                save("median_tv", grp_med["tv_list"][-1])
                save("median_compliance", grp_med["compliance_list"][-1])

                med_iters = summary.require_group("median_iters")
                for k in list(med_iters.keys()):
                    del med_iters[k]
                for name in grp_med["iters"]:
                    med_iters.create_dataset(name, data=grp_med["iters"][name][()])
                    
                print(f"\n=== Summary saved (best: {best_seed}, median: {median_seed}) ===")


def parse_mesh_list(s: str) -> List[int]:
    """Parse comma/space-separated mesh sizes, e.g. '16,32,64'."""
    if not s:
        return [16, 32, 64, 128]
    parts = [p.strip() for p in s.replace(',', ' ').split()]
    return [int(p) for p in parts]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ADMM runner (adjoint) with CLI options.")
    p.add_argument("--mesh-list", type=str, default="16,32,64,128",
                   help="Comma/space-separated mesh sizes (e.g. '16,32,64').")
    p.add_argument("--break-iter", type=str, default=None, help="Comma-separated list of max outer ADMM iterations (e.g. 50,100,200).")
    p.add_argument("--rho", type=str, default="25", help="Comma/space-separated penalty parameter.")
    p.add_argument("--alpha", type=str, default=1e-5, help="Comma/space-separated Regularization parameter")
    p.add_argument("--gamma", type=float, default=2.0, help="Gamma for acceptance threshold scaling.")
    p.add_argument("--beta", type=float, default=0.90, help="Acceptance factor (eta <= beta * tau).")
    p.add_argument("--zeta", type=float, default=0.90, help="EMA factor for tau update.")
    p.add_argument("--seed-init", type=int, default=25, help="Initial random seed.")
    p.add_argument("--use-prev", action="store_true", help="Reuse previous design variable 'a' and 'b' as init.")
    p.add_argument("--backend", type=str, default="gurobi", help="backend for integer solver (mergesplit, gurobi)")
    p.add_argument("--source_strength", type=float, default=0.01)
    p.add_argument("--vol_frac", type=float, default=0.4)
    return p


def main():
    args = build_argparser().parse_args()
    mesh_list = parse_mesh_list(args.mesh_list)

    # Parse break-iter list
    if args.break_iter is not None:
        break_iter_list = [int(x) for x in args.break_iter.split(",")]
    else:
        break_iter_list = [50] * len(mesh_list)

    # Reproducibility
    np.random.seed(args.seed_init)

    # Run trials
    for idx, (dim, break_iter) in enumerate(zip(mesh_list, break_iter_list)):
        args.break_iter = break_iter

        run_trial(dim=dim, idx=idx, params=args,)
    
    print("=== Job finished and data saved ===", flush=True)


if __name__ == "__main__":
    main()
