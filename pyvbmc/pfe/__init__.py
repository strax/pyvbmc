from abc import ABC, abstractmethod


class FailureEstimator(ABC):
    @abstractmethod
    def predict(self, x):
        """Predict the probability of failure at x."""
        ...


class OracleFailureEstimator(FailureEstimator):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def predict(self, x):
        return self.func(x)
