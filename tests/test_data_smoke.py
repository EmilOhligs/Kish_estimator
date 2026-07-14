"""Smoke tests for the local dataset (skipped if data/models aren't present, e.g. in CI)."""
import pytest

from uq_mace.data import TRAIN_SET, TEST_SET_BIG, TEST_SET_SMALL, load_energies_forces
from uq_mace.ensemble import MODELS_DIR, find_model_paths

pytestmark = pytest.mark.skipif(not TRAIN_SET.exists(), reason="data/ not present (not tracked in git)")


def test_train_set_frame_count_and_atoms():
    energies, forces = load_energies_forces(TRAIN_SET)
    assert len(energies) == 2370
    # mixed set: 1399 frames with 192 atoms (64 H2O), 971 with 189 (63 H2O)
    counts = [f.shape[0] for f in forces]
    assert counts.count(192) == 1399
    assert counts.count(189) == 971


def test_testset_big_frame_count_and_atoms():
    energies, forces = load_energies_forces(TEST_SET_BIG)
    assert len(energies) == 400
    # uniform: all frames 189 atoms (63 H2O)
    assert all(f.shape == (189, 3) for f in forces)


def test_testset_small_frame_count_and_atoms():
    energies, forces = load_energies_forces(TEST_SET_SMALL)
    assert len(energies) == 125
    # mixed set: 74 frames with 192 atoms, 51 with 189
    counts = [f.shape[0] for f in forces]
    assert counts.count(192) == 74
    assert counts.count(189) == 51


@pytest.mark.skipif(not (MODELS_DIR / "ensemble_L2c").exists(), reason="models/ not present")
def test_ensemble_l2c_has_two_members():
    assert len(find_model_paths(MODELS_DIR / "ensemble_L2c")) == 2


@pytest.mark.skipif(not (MODELS_DIR / "ensemble_L0").exists(), reason="models/ not present")
def test_ensemble_l0_has_three_members():
    assert len(find_model_paths(MODELS_DIR / "ensemble_L0")) == 3
