#!/usr/bin/env python
"""Evaluate all three pre-trained MACE ensembles against DFT reference data.

Usage:
    python scripts/evaluate_ensembles.py

Reproduces what Tobi asked for (email, 2026-07-08):
    1. RMSE of each ensemble on water_test_small.xyz (expect force RMSE ~10 meV/A
       for L0, ~5 meV/A for L2; energy RMSE lands around 0.05-0.2 meV/atom).
    2. Whether the ensemble standard deviation (sigma) of forces/energies correlates
       with the actual error - the first data point for the sigma(R) vs N_eff question.
"""
import numpy as np

from uq_mace.data import TEST_SET_SMALL, load_trajectory
from uq_mace.ensemble import MODELS_DIR
from uq_mace.evaluation import (
    energy_rmse_per_atom,
    evaluate_ensemble_members,
    force_rmse,
    local_force_sigma_and_error,
    reference_energies_forces,
    sigma_error_correlation,
)

ENSEMBLES = ["ensemble_L0", "ensemble_L0c", "ensemble_L2c"]


def main():
    frames = load_trajectory(TEST_SET_SMALL)
    e_ref, f_ref, n_atoms = reference_energies_forces(frames)

    print(f"Evaluating on {TEST_SET_SMALL.name} ({len(frames)} frames)\n")

    for name in ENSEMBLES:
        model_dir = MODELS_DIR / name
        if not model_dir.exists():
            print(f"{name}: skipped (not found)")
            continue

        result = evaluate_ensemble_members(model_dir, frames)
        energies = result["energies"]  # (n_members, n_frames)

        e_mean = energies.mean(axis=0)
        e_sigma = energies.std(axis=0, ddof=1) if energies.shape[0] > 1 else np.zeros_like(e_mean)

        f_mean = [np.mean(f, axis=0) for f in result["forces"]]

        e_rmse = energy_rmse_per_atom(e_mean, e_ref, n_atoms)
        f_rmse = force_rmse(f_mean, f_ref)

        # Normalize BOTH sigma and error per atom: test_small mixes 189- and 192-atom
        # frames, so total-energy sigma vs. per-atom error would not be a constant
        # rescaling and would distort the correlation.
        e_sigma_per_atom = e_sigma / n_atoms
        abs_error_per_atom = np.abs(e_mean - e_ref) / n_atoms
        corr = sigma_error_correlation(e_sigma_per_atom, abs_error_per_atom) if energies.shape[0] > 1 else float("nan")

        local_sigma, local_error = local_force_sigma_and_error(result["forces"], f_ref)
        local_corr = (
            sigma_error_correlation(local_sigma, local_error)
            if energies.shape[0] > 1
            else float("nan")
        )

        print(f"{name} ({energies.shape[0]} members):")
        print(f"  energy RMSE: {e_rmse:.2f} meV/atom")
        print(f"  force RMSE:  {f_rmse:.2f} meV/Angstrom")
        print(f"  corr(sigma_E/atom, |error|/atom), per frame: {corr:.3f}")
        print(f"  corr(sigma_force, |error|), per atom (n={len(local_sigma)}): {local_corr:.3f}")
        print()


if __name__ == "__main__":
    main()
