"""Loading and parsing of VASP output and ASE trajectories.

Dataset layout (see data/README.md):
    data/raw/water_train.xyz          2370 frames, mixed: 1399 frames with 192 atoms (64 H2O)
                                      and 971 frames with 189 atoms (63 H2O) - MACE training set
    data/raw/water_testset_big.xyz    400 frames, 189 atoms (63 H2O) - held-out test set
    data/raw/water_test_small.xyz     125 frames, mixed: 74 frames with 192 atoms and
                                      51 frames with 189 atoms - held-out test set (small)

Frames are extended XYZ with per-atom species/positions/forces and per-frame
energy/stress/lattice (RPBE-D3, from Hilpert & Kresse 2026).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TRAIN_SET = DATA_DIR / "water_train.xyz"
TEST_SET_BIG = DATA_DIR / "water_testset_big.xyz"
TEST_SET_SMALL = DATA_DIR / "water_test_small.xyz"


def load_trajectory(path: str | Path) -> list[Atoms]:
    """Load an ASE-readable trajectory (.xyz, .traj, .extxyz, or VASP vasprun.xml).

    Returns a list of ase.Atoms, one per frame. ase.io.read is typed as
    ``Atoms | list[Atoms]``; we normalize to a list so downstream code (and static
    type checkers) can always rely on iterating over whole frames.
    """
    from ase import Atoms
    from ase.io import read

    frames = read(path, index=":")
    if isinstance(frames, Atoms):  # single-frame file: ase returns a bare Atoms
        return [frames]
    return list(frames)


def load_energies_forces(path: str | Path):
    """Convenience loader: returns (energies, forces) arrays from an extended-XYZ file.

    energies: shape (n_frames,)
    forces:   list of length n_frames, each shape (n_atoms_i, 3). The atom count varies
              between frames: water_train.xyz and water_test_small.xyz mix 64- and
              63-molecule frames (192/189 atoms); water_testset_big.xyz is uniformly 189.
              Never assume a fixed (192, 3) force shape.
    """
    import numpy as np

    frames = load_trajectory(path)
    energies = np.array([f.get_potential_energy() for f in frames])
    forces = [f.get_forces() for f in frames]
    return energies, forces
