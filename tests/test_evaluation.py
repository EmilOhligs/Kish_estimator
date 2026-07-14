import numpy as np

from uq_mace.evaluation import energy_rmse_per_atom, force_rmse, sigma_error_correlation


def test_energy_rmse_per_atom_zero_when_exact():
    e_pred = np.array([-945.0, -940.0])
    n_atoms = np.array([192, 192])
    assert energy_rmse_per_atom(e_pred, e_pred, n_atoms) == 0.0


def test_energy_rmse_per_atom_known_value():
    # single frame, 100 atoms, 1 eV total error -> 10 meV/atom
    e_pred = np.array([1.0])
    e_ref = np.array([0.0])
    n_atoms = np.array([100])
    assert np.isclose(energy_rmse_per_atom(e_pred, e_ref, n_atoms), 10.0)


def test_force_rmse_zero_when_exact():
    f_pred = [np.ones((5, 3)), np.zeros((3, 3))]
    assert force_rmse(f_pred, f_pred) == 0.0


def test_force_rmse_known_value():
    # constant offset of 0.01 eV/Angstrom in every component -> 10 meV/Angstrom RMSE
    f_ref = [np.zeros((4, 3))]
    f_pred = [np.full((4, 3), 0.01)]
    assert np.isclose(force_rmse(f_pred, f_ref), 10.0)


def test_sigma_error_correlation_perfect_positive():
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    error = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(sigma_error_correlation(sigma, error), 1.0)


def test_sigma_error_correlation_no_relationship():
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    error = np.array([1.0, 2.0, 3.0, 4.0])
    # constant sigma -> correlation is undefined (NaN); just check it doesn't crash
    result = sigma_error_correlation(sigma, error)
    assert np.isnan(result) or np.isfinite(result)


def test_local_force_sigma_and_error_shapes_and_values():
    from uq_mace.evaluation import local_force_sigma_and_error

    # 2 frames, 3 members, frame 0 has 2 atoms, frame 1 has 1 atom
    forces_stacked = [
        np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-0.02, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]),  # shape (3 members, 2 atoms, 3)
        np.array([
            [[5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
        ]),  # shape (3 members, 1 atom, 3)
    ]
    f_ref = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[5.1, 0.0, 0.0]]),
    ]

    sigma, error = local_force_sigma_and_error(forces_stacked, f_ref)
    assert sigma.shape == (3,)
    assert error.shape == (3,)
    # second atom (index 1) and third atom (frame 2): zero ensemble spread -> sigma == 0
    assert np.isclose(sigma[1], 0.0)
    assert np.isclose(sigma[2], 0.0)
    # frame 2's atom has a constant 0.1 eV/Angstrom offset from reference
    assert np.isclose(error[2], 0.1)
