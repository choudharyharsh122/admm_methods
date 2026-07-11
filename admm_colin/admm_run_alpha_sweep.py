"""
The file just takes a list of alpha values and rho values for running the admm algorithm
Internally this file just assembles that list and calls admm_run.py which is the main file
admm_run.py is a standalone file which can also be run separately for a single configuration
All the configuration parameters are supplied from admm_config.cfg
This file is just used to run an alpha sweep for comparison with Kernel method
Otherwise, to run a single test just run $ python admm_run.py

For the generated tests, range of alpha values were used (from 0.01 to 1.0) and rho was kept constant at 5
"""


import argparse
import copy
import itertools
from typing import Iterable, List, Tuple

import numpy as np

from admm_run import CONFIG_FILE, load_config, run_trial


def _parse_float_list(raw: str) -> List[float]:
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    if not tokens:
        raise ValueError("Empty list provided. Pass values like '0.01,0.02' or '0.01 0.02'.")
    return [float(t) for t in tokens]


def _build_combinations(
    alpha_list: Iterable[float], rho_list: Iterable[float], mode: str
) -> List[Tuple[float, float]]:
    alpha_list = list(alpha_list)
    rho_list = list(rho_list)

    if mode == "zip":
        if len(alpha_list) != len(rho_list):
            raise ValueError(
                "For pairing='zip', alpha-list and rho-list must have the same length."
            )
        return list(zip(alpha_list, rho_list))

    return list(itertools.product(alpha_list, rho_list))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run admm_run for a list of alpha and rho values."
    )
    parser.add_argument(
        "--alpha-list",
        required=True,
        help="Comma/space-separated alpha values. Example: '0.01,0.02,0.05'",
    )
    parser.add_argument(
        "--rho-list",
        required=True,
        help="Comma/space-separated rho values. Example: '1,5,10'",
    )
    parser.add_argument(
        "--pairing",
        choices=["cartesian", "zip"],
        default="cartesian",
        help=(
            "How to pair alpha and rho lists: "
            "'cartesian' runs all combinations, 'zip' runs pairwise."
        ),
    )

    cli_args = parser.parse_args()

    alpha_values = _parse_float_list(cli_args.alpha_list)
    rho_values = _parse_float_list(cli_args.rho_list)
    pairs = _build_combinations(alpha_values, rho_values, cli_args.pairing)

    config = load_config(CONFIG_FILE)
    base_args = argparse.Namespace(**config)

    # Keep reproducibility behavior consistent with admm_run.py.
    np.random.seed(base_args.SEED_INIT)

    print(f"=== Starting batch run with {len(pairs)} (alpha, rho) settings ===", flush=True)

    for idx, (alpha, rho) in enumerate(pairs):
        run_args = copy.deepcopy(base_args)
        run_args.ALPHA = float(alpha)
        run_args.RHO = float(rho)

        print(
            f"\n=== Batch item {idx + 1}/{len(pairs)}: ALPHA={run_args.ALPHA}, RHO={run_args.RHO} ===",
            flush=True,
        )

        run_trial(dim=int(run_args.MESH_SIZE), idx=idx, params=run_args)

    print("=== Batch job finished and data saved ===", flush=True)


if __name__ == "__main__":
    main()