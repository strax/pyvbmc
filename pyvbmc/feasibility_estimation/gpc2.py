from __future__ import annotations

import logging
import time
from typing import Callable

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from torchmin import Minimizer

from pyvbmc.function_logger import FunctionLogger

from . import FeasibilityEstimator
from .utils import get_evaluations

logger = logging.getLogger(__name__)


def _as_tensor(x: Tensor | NDArray) -> Tensor:
    if not isinstance(x, Tensor):
        x = torch.from_numpy(x)
    return x


def _approx_sigmoid_gaussian_conv(mu: Tensor, sigma2: Tensor) -> Tensor:
    return torch.sigmoid(mu / torch.sqrt(1 + torch.pi / 8 * sigma2))


class BinaryDirichletGPC(ExactGP):
    def __init__(self, X: Tensor, y: Tensor):
        y = y.to(torch.int)
        likelihood = DirichletClassificationLikelihood(y, dtype=torch.double)
        assert likelihood.num_classes == 2
        super().__init__(X, likelihood.transformed_targets, likelihood)
        self.likelihood = likelihood
        batch_shape = torch.Size((2,))
        self.mean = ConstantMean(batch_shape=batch_shape).double()
        self.cov = ScaleKernel(
            MaternKernel(5 / 2, batch_shape=batch_shape).double(),
            batch_shape=batch_shape,
        ).double()

    def forward(self, x: Tensor):
        mean, cov = self.mean(x), self.cov(x)
        return MultivariateNormal(mean, cov)

    def set_train_data(self, inputs: Tensor, targets: Tensor):
        X, y = inputs, targets.to(torch.int)

        self.likelihood = DirichletClassificationLikelihood(
            y, dtype=torch.double
        )
        super().set_train_data(
            X, self.likelihood.transformed_targets, strict=False
        )


class GPCFeasibilityEstimator(FeasibilityEstimator):
    X: Tensor = torch.empty(0)
    y: Tensor = torch.empty(0)
    model: BinaryDirichletGPC | None = None
    optimize_after_update: bool
    fast_predictive_integration: bool

    def __init__(
        self, *, optimize_after_update=False, fast_predictive_integration=True
    ):
        self.optimize_after_update = optimize_after_update
        self.fast_predictive_integration = fast_predictive_integration

    def _init_model(self):
        if not (0 < self.y.count_nonzero() < self.y.numel()):
            logger.debug(
                "Skipping model initialization due to not having seen both failed/succeeded observations"
            )
            return
        logger.debug("(Re)initializing model")
        self.model = BinaryDirichletGPC(self.X, self.y)
        self._optimize_hyperparameters()
        return self.model

    def _optimize_hyperparameters(self):
        assert self.model is not None
        model, likelihood = self.model, self.model.likelihood

        optimizer = Minimizer(model.hyperparameters(), max_iter=1000, tol=1e-8)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        def step():
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            objective = -mll(output, model.train_targets).sum()
            return objective

        time_begin = time.monotonic()
        loss = optimizer.step(step)
        time_end = time.monotonic()
        logger.debug(
            "Optimized hyperparameters in %.4fs, MLL = %.6f",
            time_end - time_begin,
            loss,
        )
        assert torch.all(
            torch.isfinite(loss)
        ), "Optimization resulted in nonfinite parameters"

        model.eval()
        likelihood.eval()

    @torch.inference_mode
    def predict(self, x: NDArray):
        if self.model is None:
            *batch_dims, _ = np.shape(np.atleast_2d(x))
            return np.broadcast_to(1.0, batch_dims).squeeze()

        x = _as_tensor(np.atleast_2d(x)).double()
        with gpytorch.settings.fast_computations(False, False, False):
            predictive = self.model(x)
            # Approximate eq. 8, either with a known good approximation or MC
            if self.fast_predictive_integration:
                mu = predictive.mean[0] - predictive.mean[1]
                sigma2 = predictive.variance[0] + predictive.variance[1]
                p_failure = _approx_sigmoid_gaussian_conv(mu, sigma2)
            else:
                p_failure, _ = (
                    predictive.sample(torch.Size((256,))).softmax(1).mean(0)
                )
            return 1.0 - p_failure.numpy(force=True).squeeze()

    def update(
        self, x: NDArray, y: NDArray, *, function_logger: FunctionLogger
    ):
        del x, y

        if function_logger.Xn < 2:
            return

        X, y = get_evaluations(function_logger)
        X = torch.from_numpy(X).double()
        y = torch.from_numpy(y).isfinite().squeeze_()

        if self.y is not None and torch.numel(y) == torch.numel(self.y):
            logger.debug(
                "Update was requested more than once for the same observation set, skipping posterior update"
            )
            return
        self.X = X
        self.y = y

        if self.model is None or self.optimize_after_update:
            logger.debug("Update caused model to be reinitialized")
            self._init_model()
        else:
            n_succeeded = torch.count_nonzero(self.y)
            n_failed = torch.numel(self.y) - n_succeeded
            logger.debug(
                "Updating GPC posterior (succeeded: %s, failed: %s)",
                n_succeeded.item(),
                n_failed.item(),
            )

            # TODO: Use `self.model.get_fantasy_model` if/when it is fixed
            self.model.set_train_data(self.X, self.y)

    def optimize(self):
        # If `optimize_after_update` is set, the hyperparameters are already optimized
        if not self.optimize_after_update:
            # Recreating the whole model is a bit inefficient, but
            # hyperparameter optimization does not work if resumed from a previous state
            self._init_model()


__all__ = ["GPCFeasibilityEstimator"]
