from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from pyvbmc.function_logger import FunctionLogger


class FeasibilityEstimator(Protocol):
    @abstractmethod
    def prob(self, x: ArrayLike) -> ArrayLike:
        """Return the estimated probability of feasibility at x."""
        raise NotImplementedError

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        """Return the estimated log-probability of feasibility at x."""
        p = self.prob(x)
        with np.errstate(divide="ignore"):
            return np.log(p)

    def update(self, x, y, *, function_logger: FunctionLogger):
        """Update the estimator with an observation f(x) = y."""

    def optimize(self):
        pass


class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, prob):
        super().__init__()
        self._prob = prob

    def prob(self, x):
        return self._prob(x)
