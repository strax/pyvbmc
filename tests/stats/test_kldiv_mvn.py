import numpy as np
from pyvbmc.stats import kldiv_mvn


def test_kldiv_1d_identical():
    mu = np.ones((1, 1))
    sigma = np.ones((1, 1))
    assert np.all(kldiv_mvn(mu, sigma, mu, sigma) == 0)


def test_kldiv_1d_seperated():
    mu = np.ones((1, 1))
    sigma = np.ones((1, 1))
    assert np.all(kldiv_mvn(mu, sigma, mu * 100, sigma) == 4900.5)


def test_kldiv_3d_identical():
    mu = np.ones((3, 1))
    sigma = np.eye(3)
    assert np.all(kldiv_mvn(mu, sigma, mu, sigma) == 0)


def test_kldiv_3d_seperated():
    mu = np.ones((3, 1))
    sigma = np.eye(3)
    assert np.all(kldiv_mvn(mu, sigma, mu * 100, sigma) == 14701.5)
