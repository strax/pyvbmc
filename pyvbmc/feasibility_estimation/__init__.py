from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvbmc.function_logger import FunctionLogger


class FeasibilityEstimator(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict the probability of feasibility at x."""

    def update(self, x, y, *, function_logger: FunctionLogger):
        """Update the estimator with an observation f(x) = y."""

    def optimize(self):
        pass


class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def predict(self, x):
        return self.prob(x)
