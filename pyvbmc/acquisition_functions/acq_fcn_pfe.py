import sys

import gpyreg as gpr
import numpy as np
from scipy.special import log1p

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .acq_fcn_log import AcqFcnLog


class AcqFcnFailureRobustLog(AcqFcnLog):
    """
    Acquisition function for prospective uncertainty search (log-valued) with failure estimation.
    """

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def _compute_acquisition_function(
        self,
        Xs: np.ndarray,
        vp: VariationalPosterior,
        gp: gpr.GP,
        function_logger: FunctionLogger,
        optim_state: dict,
        f_mu: np.ndarray,
        f_s2: np.ndarray,
        f_bar: np.ndarray,
        var_tot: np.ndarray,
    ):
        """
        Compute the value of the acquisition function.
        """
        # Xs is in *transformed* coordinates
        acq = super()._compute_acquisition_function(
            Xs,
            vp,
            gp,
            function_logger,
            optim_state,
            f_mu,
            f_s2,
            f_bar,
            var_tot,
        )

        Xs_orig = function_logger.parameter_transformer.inverse(Xs)
        epf = np.clip(
            [self.estimator.predict(x_orig) for x_orig in Xs_orig], 0, 1
        )
        out = acq + log1p(-epf)
        if np.any(np.isnan(out)):
            breakpoint()
            raise RuntimeError("Acquisition value is NaN")
        return out
