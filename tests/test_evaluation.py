import numpy as np
import pytest
from inf8239_u03_gen.evaluation import (
    split_indices,
    kl_numpy,
    interpolation_points,
    nearest_mse,
    diversity_mse,
    mse_per_image,
)


def test_split_disjoint_and_reproducible():
    a, b = split_indices()
    a2, b2 = split_indices()
    assert len(a) == 54000 and len(b) == 6000
    assert not set(a) & set(b) and len(set(a) | set(b)) == 60000
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_kl_standard_normal_zero():
    assert np.allclose(kl_numpy(np.zeros((3, 2)), np.zeros((3, 2))), 0)


def test_kl_known_example():
    assert kl_numpy(np.array([[1.0]]), np.log([[0.25]]))[0] == pytest.approx(
        0.81814718056
    )


def test_interpolation_endpoints():
    p = interpolation_points([-2, 1], [2, 3])
    assert np.allclose(p[0], [-2, 1]) and np.allclose(p[-1], [2, 3])


def test_nearest_exact_and_known_distance():
    refs = np.array([[[[0.0]]], [[[1.0]]]], dtype="float32")
    q = np.array([[[[0.1]]], [[[1.0]]]], dtype="float32")
    idx, d = nearest_mse(q, refs)
    assert idx.tolist() == [0, 1]
    assert np.allclose(d, [0.01, 0], atol=1e-6)


def test_diversity_is_pairwise_average():
    x = np.array([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1)
    assert diversity_mse(x) == pytest.approx((1 + 4 + 1) / 3)


def test_mse_known_example():
    assert mse_per_image(
        np.array([1.0, 0.0]).reshape(1, 1, 2, 1),
        np.array([0.8, 0.1]).reshape(1, 1, 2, 1),
    )[0] == pytest.approx(0.025)
