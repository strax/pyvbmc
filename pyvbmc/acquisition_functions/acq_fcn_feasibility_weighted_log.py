import sys

import gpyreg as gpr
import numpy as np

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.timer import main_timer as timer

from .acq_fcn_log import AcqFcnLog


class AcqFcnFeasibilityWeightedLog(AcqFcnLog):
    """
    Acquisition function for feasibility-weighted prospective uncertainty search
    (log-valued).
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

        timer.start_timer("fe_predict")
        p_feasible = self.estimator.log_prob(Xs_orig)
        timer.stop_timer("fe_predict")

        out = acq - p_feasible
        return out
