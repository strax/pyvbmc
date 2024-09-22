import numpy as np

from pyvbmc.function_logger import FunctionLogger
from typing import Tuple
from numpy.typing import NDArray


def get_evaluations(
    function_logger: FunctionLogger,
) -> Tuple[NDArray, NDArray]:
    mask = np.all(~np.isnan(function_logger.y_orig), axis=-1)
    X = function_logger.X_orig[mask]
    y = function_logger.y_orig[mask]
    return X, y


__all__ = ["get_evaluations"]
