"""Evaluate MACE ensembles against DFT reference data.

Implements the following evaluation protocol:
    - load an ensemble of MACE models with ASE (mace.calculators.MACECalculator)
    - attach each model as atoms.calc, read off atoms.get_potential_energy() /
      atoms.get_forces()
    - compare against the original DFT energy/forces stored in the extended-XYZ files
    - compute energy RMSE (meV/atom) and force RMSE (meV/Angstrom)
    - compute the ensemble spread (std across members) per configuration and check
      whether it correlates with the actual error (the core UQ question of this project)

Expected benchmarks on water_test_small.xyz (project reference values, 2026-07; note these are
FORCE RMSEs in meV/Angstrom - energy RMSE comes out around 0.05-0.2 meV/atom):
    L0 models: force RMSE ~10 meV/Angstrom
    L2 models: force RMSE ~5 meV/Angstrom
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure metrics (no torch/mace dependency - safe to unit test everywhere)
# ---------------------------------------------------------------------------

def energy_rmse_per_atom(e_pred: np.ndarray, e_ref: np.ndarray, n_atoms: np.ndarray) -> float:
    """RMSE of per-atom energy, in meV/atom.

    e_pred, e_ref: shape (n_frames,), total energy per frame, in eV.
    n_atoms: shape (n_frames,), atom count per frame (can vary between frames).
    """
    errors_per_atom = (np.asarray(e_pred) - np.asarray(e_ref)) / np.asarray(n_atoms)
    return float(np.sqrt(np.mean(errors_per_atom ** 2)) * 1000.0)


def force_rmse(f_pred: list[np.ndarray], f_ref: list[np.ndarray]) -> float:
    """RMSE of force components, in meV/Angstrom.

    f_pred, f_ref: lists of length n_frames, each entry shape (n_atoms_i, 3), in eV/Angstrom.
    """
    diffs = np.concatenate([(fp - fr).ravel() for fp, fr in zip(f_pred, f_ref)])
    return float(np.sqrt(np.mean(diffs ** 2)) * 1000.0)


def sigma_error_correlation(sigma: np.ndarray, abs_error: np.ndarray) -> float:
    """Pearson correlation between the ensemble spread sigma(R) and the actual |error|.

    This is the central diagnostic for the project: does a larger ensemble
    disagreement predict a larger true error?
    """
    sigma = np.asarray(sigma)
    abs_error = np.asarray(abs_error)
    return float(np.corrcoef(sigma, abs_error)[0, 1])


# ---------------------------------------------------------------------------
# Ensemble evaluation (requires mace-torch + torch to be installed)
# ---------------------------------------------------------------------------

def evaluate_ensemble_members(model_dir: str | Path, frames, device: str = "cpu"):
    """Run every individual model in model_dir on every frame.

    Returns a dict with:
        energies: array (n_members, n_frames) - predicted total energy per frame, eV
        forces:   list length n_frames, each array (n_members, n_atoms_i, 3), eV/Angstrom

    Frames must already have their original DFT energy/forces readable via
    frame.info / frame.arrays (e.g. straight from ase.io.read of an extended-XYZ file) -
    this function only overwrites frame.calc, it does not touch the stored reference data.
    """
    from mace.calculators import MACECalculator

    from .ensemble import find_model_paths

    model_paths = find_model_paths(model_dir)
    n_members = len(model_paths)
    n_frames = len(frames)

    energies = np.zeros((n_members, n_frames))
    forces: list[list[np.ndarray]] = [[] for _ in range(n_frames)]

    for m, path in enumerate(model_paths):
        calc = MACECalculator(model_paths=[path], device=device)
        for i, atoms in enumerate(frames):
            atoms.calc = calc
            energies[m, i] = atoms.get_potential_energy()
            forces[i].append(atoms.get_forces())

    forces_stacked = [np.stack(f_list, axis=0) for f_list in forces]  # (n_members, n_atoms_i, 3)
    return {"energies": energies, "forces": forces_stacked}


def local_force_sigma_and_error(forces_stacked: list[np.ndarray], f_ref: list[np.ndarray]):
    """Pool per-atom force sigma and per-atom force error across all frames.

    This is the "local average" variant: instead of one sigma/error pair per
    frame (coarse, ~125 points), get one pair per atom (fine, ~125 x 190 points).

    forces_stacked: list length n_frames, each array (n_members, n_atoms_i, 3) - as
                     returned by evaluate_ensemble_members()["forces"].
    f_ref: list length n_frames, each array (n_atoms_i, 3) - DFT reference forces.

    Returns (sigma_per_atom, error_per_atom): both 1D, pooled over all atoms and frames.
        sigma_per_atom: std across ensemble members of the force vector (averaged over
                        the 3 components), per atom.
        error_per_atom: |mean-prediction force vector - DFT force vector|, per atom.
    """
    sigmas = []
    errors = []
    for f_members, f_r in zip(forces_stacked, f_ref):
        f_mean = f_members.mean(axis=0)  # (n_atoms, 3)
        if f_members.shape[0] > 1:
            per_atom_sigma = f_members.std(axis=0, ddof=1).mean(axis=-1)  # (n_atoms,)
        else:
            per_atom_sigma = np.zeros(f_mean.shape[0])
        per_atom_error = np.linalg.norm(f_mean - f_r, axis=-1)  # (n_atoms,)
        sigmas.append(per_atom_sigma)
        errors.append(per_atom_error)
    return np.concatenate(sigmas), np.concatenate(errors)


def reference_energies_forces(frames):
    """Extract the original DFT energy/forces stored in each frame, before any
    calculator is attached (call this BEFORE evaluate_ensemble_members, which
    overwrites frame.calc)."""
    e_ref = np.array([f.get_potential_energy() for f in frames])
    f_ref = [f.get_forces() for f in frames]
    n_atoms = np.array([len(f) for f in frames])
    return e_ref, f_ref, n_atoms
