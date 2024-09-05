import numpy as np
from numpy.testing import assert_equal
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.utils.validation import NotFittedError, check_is_fitted

from pyvbmc.function_logger import FunctionLogger

from . import FeasibilityEstimator


class GPCFeasibilityEstimator(FeasibilityEstimator):
    _gpc: GaussianProcessClassifier

    def __init__(self):
        self._gpc = GaussianProcessClassifier()

    @property
    def _is_fitted(self) -> bool:
        try:
            check_is_fitted(self._gpc)
            return True
        except NotFittedError:
            return False

    def update(self, x, y, function_logger: FunctionLogger):
        del x, y

        if function_logger.Xn < 2:
            return
        mask = np.all(~np.isnan(function_logger.y_orig), axis=-1)
        X = function_logger.X_orig[mask]
        y = function_logger.y_orig[mask]

        # 1 = feasible, 0 = infeasible
        f = np.where(np.isfinite(y), 1, 0).ravel()

        # Check if we have seen both feasible and infeasible points as
        # GaussianProcessClassifier requires both to be present for fitting
        if 0 < np.count_nonzero(f) < np.size(f):
            # print("fitting gpc")
            self._gpc.fit(X, f)
            # FIXME: Remove this check
            assert_equal(np.array(self._gpc.classes_), np.array([0, 1]))

    def predict(self, x):
        if self._is_fitted:
            p_feasible = self._gpc.predict_proba(np.atleast_2d(x))[
                :, 1
            ].squeeze()
            return p_feasible
        else:
            return 1.0
