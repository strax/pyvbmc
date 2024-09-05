from abc import ABC, abstractmethod

import numpy as np


class FeasibilityEstimator(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict the probability of feasibility at x."""

    def update(self, x, y):
        """Update the estimator with an observation f(x) = y."""


class OracleFeasibilityEstimator(FeasibilityEstimator):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def predict(self, x):
        return self.prob(x)
