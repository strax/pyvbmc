from __future__ import annotations

import logging
from time import time
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

from pyvbmc.function_logger import FunctionLogger

from . import FeasibilityEstimator

logger = logging.getLogger(__name__)


def _optimize(
    step: Callable[[], float],
    optimizer: Optimizer,
    *,
    max_iter=1000,
    ftol=1e-6,
) -> tuple[float, int]:
    prev_objective = torch.inf
    for i in range(max_iter):
        objective = optimizer.step(step)
        if prev_objective - objective < ftol:
            logger.debug(
                "Early stopping: %s < %s", prev_objective - objective, ftol
            )
            break
        prev_objective = objective
    return objective, i + 1


def _convert_to_tensor(
    input: Tensor | NDArray, *, dtype=torch.float
) -> Tensor:
    if not isinstance(input, Tensor):
        input = torch.from_numpy(input)
    return input.to(dtype)


class BinaryDirichletGPC(ExactGP):
    def __init__(self, X: Tensor, y: Tensor):
        y = y.to(torch.int)
        likelihood = DirichletClassificationLikelihood(y)
        assert likelihood.num_classes == 2
        super().__init__(X, likelihood.transformed_targets, likelihood)
        self.likelihood = likelihood
        batch_shape = torch.Size((2,))
        self.mean = ConstantMean(batch_shape=batch_shape)
        self.cov = ScaleKernel(
            MaternKernel(5 / 2, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x: Tensor):
        mean, cov = self.mean(x), self.cov(x)
        return MultivariateNormal(mean, cov)

    def set_train_data(self, inputs: Tensor, targets: Tensor):
        X, y = inputs, targets.to(torch.int)

        self.likelihood = DirichletClassificationLikelihood(y)
        super().set_train_data(
            X, self.likelihood.transformed_targets, strict=False
        )


class GPCFeasibilityEstimator(FeasibilityEstimator):
    X: Tensor = torch.empty(0)
    y: Tensor = torch.empty(0)
    model: BinaryDirichletGPC | None = None
    optimize_after_update: bool
    use_fast_integrator: bool

    def __init__(
        self, *, optimize_after_update=False, use_fast_integrator=True
    ):
        self.optimize_after_update = optimize_after_update
        self.use_fast_integrator = use_fast_integrator

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

    def _optimize_hyperparameters(self, **kwargs):
        assert self.model is not None
        model, likelihood = self.model, self.model.likelihood

        optimizer = LBFGS(model.hyperparameters(), max_iter=1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()

        def step():
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            objective = -mll(output, model.train_targets).sum()
            objective.backward()
            return objective.item()

        time_begin = time()
        loss, iters = _optimize(step, optimizer, **kwargs)
        time_elapsed = time() - time_begin
        logger.debug(
            "Optimized hyperparameters in %.2fs (%d iterations), MLL: %.4f",
            time_elapsed,
            iters,
            loss,
        )

        model.eval()
        likelihood.eval()

    @torch.no_grad
    def predict(self, x: NDArray):
        if self.model is None:
            *batch_dims, _ = np.shape(np.atleast_2d(x))
            return np.broadcast_to(1.0, batch_dims).squeeze()

        x = _convert_to_tensor(np.atleast_2d(x))
        with gpytorch.settings.fast_computations():
            predictive = self.model(x)
            # Approximate eq. 8, either with a known good approximation or MC
            if self.use_fast_integrator:
                f = predictive.mean[0] - predictive.mean[1]
                v = predictive.variance[0] + predictive.variance[1]
                p_failure = torch.sigmoid(f / torch.sqrt(1 + torch.pi / 8 * v))
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

        mask = np.all(~np.isnan(function_logger.y_orig), axis=-1)
        X = torch.from_numpy(function_logger.X_orig[mask]).float()
        y = torch.from_numpy(
            np.isfinite(function_logger.y_orig[mask])
        ).squeeze()

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
