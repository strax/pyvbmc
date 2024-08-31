import numpy as np

from abc import ABC, abstractmethod

from pyvbmc.function_logger import FunctionLogger


class FailureEstimator(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict the probability of failure at x."""
        ...

    def update(self, x, y):
        """Update the estimator with an observation f(x) = y."""


class OracleFailureEstimator(FailureEstimator):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def predict(self, x):
        return self.func(x)
