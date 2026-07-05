"""Shared test fixtures: float64 keras, seeded RNG, sample populations."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import pytest

keras.backend.set_floatx("float64")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def f_batch(rng):
    """Random positive mass-normalised D2Q9 populations, shape (32, 9)."""
    f = np.abs(rng.normal(1.0, 0.2, size=(32, 9)))
    return f / f.sum(axis=1, keepdims=True)


@pytest.fixture(scope="session")
def f_pred(rng):
    """A second normalised batch playing the role of a raw NN prediction."""
    f = np.abs(rng.normal(1.0, 0.3, size=(32, 9)))
    return f / f.sum(axis=1, keepdims=True)
