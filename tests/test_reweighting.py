import numpy as np

from uq_mace.reweighting import effective_sample_size, reweighting_weights, reweighted_average


def test_effective_sample_size_uniform_weights():
    # equal weights -> N_eff should equal N
    w = np.ones(10)
    assert np.isclose(effective_sample_size(w), 10.0)


def test_effective_sample_size_single_dominant_weight():
    w = np.zeros(10)
    w[0] = 1.0
    assert np.isclose(effective_sample_size(w), 1.0)


def test_reweighting_weights_zero_delta_gives_uniform():
    e_dft = np.array([1.0, 2.0, 3.0])
    e_model = e_dft.copy()
    w = reweighting_weights(e_dft, e_model, beta=1.0)
    assert np.allclose(w, w[0])


def test_reweighted_average_matches_plain_average_for_uniform_weights():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert np.isclose(reweighted_average(a, w), a.mean())
