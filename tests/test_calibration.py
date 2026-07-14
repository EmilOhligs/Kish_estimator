import numpy as np

from uq_mace.calibration import empirical_coverage, calibration_error


def test_empirical_coverage_all_within_one_sigma():
    errors = np.array([0.1, -0.2, 0.05])
    sigmas = np.array([1.0, 1.0, 1.0])
    assert empirical_coverage(errors, sigmas, z=1.0) == 1.0


def test_calibration_error_zero_when_matching_target():
    errors = np.array([0.0] * 68 + [10.0] * 32)
    sigmas = np.ones(100)
    err = calibration_error(errors, sigmas, z=1.0, target=0.68)
    assert err < 1e-9
