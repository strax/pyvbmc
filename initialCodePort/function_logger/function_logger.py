import numpy as np

from parameter_transformer import ParameterTransformer
from timer import Timer


class FunctionLogger(object):
    """
    FunctionLogger Evaluates a function and caches results

    Porting status
    --------------
     - in _record(...) all the part
     about checking (and dealing with) identical inputs
     is missing
    """

    def __init__(
        self,
        fun,
        nvars: int,
        noise_flag: bool,
        uncertainty_handling_level: int,
        cache_size: int = 500,
        parameter_transformer: ParameterTransformer = None,
    ):
        """
        __init__ [summary]

        Parameters
        ----------
        fun : callable
            The function to be logged
            fun must take a vector input and return a scalar value and,
            optionally, the (estimated) SD of the returned value (if the
            function fun is stochastic)
        nvars : int
            number of dimensions that the function takes as input
        noise_flag : bool
            whether the function fun is stochastic
        uncertainty_handling_level : int
            uncertainty handling level
            (0: none; 1: unknown noise level; 2: user-provided noise)
        cache_size : int, optional
            initial size of caching table (default 500)
        parameter_transformer : ParameterTransformer, optional
            required to transform the parameters between
            constrained and unconstrained space, by default None
        """
        self.fun = fun
        self.nvars: int = nvars
        self.noise_flag: bool = noise_flag
        self.uncertainty_handling_level: int = uncertainty_handling_level
        self.transform_parameters = parameter_transformer is not None
        self.parameter_transformer = parameter_transformer

        self.func_count: int = 0
        self.cache_count: int = 0
        self.x_orig = np.empty((cache_size, self.nvars))
        self.y_orig = np.empty((cache_size, 1))
        self.x = np.empty((cache_size, self.nvars))
        self.y = np.empty((cache_size, 1))
        self.nevals = np.zeros((cache_size, 1))

        if self.noise_flag:
            self.S = np.empty((cache_size, 1))

        self.Xn: int = -1  # Last filled entry
        self.X_flag = np.full((cache_size, 1), False, dtype=bool)
        self.y_max = float("-Inf")
        self.fun_evaltime = np.zeros((cache_size, 1))
        self.total_fun_evaltime = 0

    def __call__(self, x: np.ndarray):
        """
        __call__ evaluates function FUN at x and caches values

        Parameters
        ----------
        x : np.ndarray
            The point at which the function will be evaluated

        Returns
        -------
        (float, float, int)
            result of the evaluatation and
            optionally the (estimated) SD of the returned value
            and the index of the last updated entry
        """

        timer = Timer()

        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        try:
            timer.start_timer("funtime")
            if self.noise_flag and self.uncertainty_handling_level == 2:
                fval_orig, fsd = self.fun(x_orig)
            else:
                fval_orig = self.fun(x_orig)
                if self.noise_flag:
                    fsd = 1
                else:
                    fsd = None
            timer.stop_timer("funtime")

        except Exception as err:
            err.args += ((
                "FunctionLogger:FuncError "
                + "Error in executing the logged function"
                + "with input: "
                + str(x_orig), )
            )
            raise

        # Check function value
        if np.any(
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            raise ValueError(
                "FunctionLogger:InvalidFuncValue"
                + "The returned function value must be a finite real-valued scalar"
                + "(returned value: "
                + str(fval_orig)
                + ")"
            )

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd) or not np.isfinite(fsd) or not np.isreal(fsd)
        ):
            raise ValueError(
                "FunctionLogger:InvalidNoiseValue"
                + "The returned estimated SD (second function output)"
                + "must be a finite, positive real-valued scalar (returned SD: "
                + str(fsd)
                + ")."
            )

        # record timer stats
        funtime = timer.get_duration("funtime")

        self.func_count += 1
        fval, idx = self._record(x_orig, x, fval_orig, fsd, funtime)

        # optimstate.N = self.Xn
        # optimstate.Neff = np.sum(self.nevals[self.X_flag])
        # optimState.totalfunevaltime = optimState.totalfunevaltime + t;
        return fval, fsd, idx

    def add(self, x: np.ndarray, fval_orig: float, fsd: float = None):
        """
        add Add previously evaluated function sample

        Parameters
        ----------
        x : np.ndarray
            the point at which the function has been evaluated
        fval_orig : float
            the result of the evaluatation
        fsd : float, optional
            (estimated) SD of the returned value
            (if heteroskedastic noise handling is on), by default None

        Returns
        -------
        None
            ???
        """
        # Convert back to original space
        if self.transform_parameters:
            x_orig = self.parameter_transformer.inverse(
                np.reshape(x, (1, x.shape[0]))
            )[0]
        else:
            x_orig = x

        if self.noise_flag:
            if fsd is None:
                fsd = 1
        else:
            fsd = None

        # Check function value
        if (
            not np.isscalar(fval_orig)
            or not np.isfinite(fval_orig)
            or not np.isreal(fval_orig)
        ):
            raise ValueError(
                "FunctionLogger:InvalidFuncValue"
                + "The returned function value must be a finite real-valued scalar"
                + "(returned value: "
                + str(fval_orig)
                + ")"
            )

        # Check returned function SD
        if self.noise_flag and (
            not np.isscalar(fsd) or not np.isfinite(fsd) or not np.isreal(fsd)
        ):
            raise ValueError(
                "FunctionLogger:InvalidNoiseValue"
                + "The returned estimated SD (second function output)"
                + "must be a finite, positive real-valued scalar (returned SD: "
                + str(fsd)
                + ")."
            )

        self.cache_count += 1
        fval, idx = self._record(x_orig, x, fval_orig, fsd, None)

    def finalize(self):
        """
        finalize remove unused caching entries
        """
        self.x_orig = self.x_orig[: self.Xn + 1]
        self.y_orig = self.y_orig[: self.Xn + 1]

        # in the original matlab version X and Y get deleted
        self.x = self.x[: self.Xn + 1]
        self.y = self.y[: self.Xn + 1]

        if self.noise_flag:
            self.S = self.S[: self.Xn + 1]
        self.X_flag = self.X_flag[: self.Xn + 1]
        self.fun_evaltime = self.fun_evaltime[: self.Xn + 1]

    def _expand_arrays(self, resize_amount: int = None):
        """
        _expand_arrays a private function to extend the rows of the object attribute arrays

        Parameters
        ----------
        resize_amount : int, optional
            additional rows, by default expand current table by 50%
        """

        if resize_amount is None:
            resize_amount = int(np.max((np.ceil(self.Xn / 2), 1)))

        self.x_orig = np.append(
            self.x_orig, np.empty((resize_amount, self.nvars)), axis=0
        )
        self.y_orig = np.append(
            self.y_orig, np.empty((resize_amount, 1)), axis=0
        )
        self.x = np.append(
            self.x, np.empty((resize_amount, self.nvars)), axis=0
        )
        self.y = np.append(self.y, np.empty((resize_amount, 1)), axis=0)

        if self.noise_flag:
            self.S = np.append(self.S, np.empty((resize_amount, 1)), axis=0)
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount, 1), True, dtype=bool), axis=0
        )
        self.fun_evaltime = np.append(
            self.fun_evaltime, np.empty((resize_amount, 1)), axis=0
        )
        self.nevals = np.append(
            self.nevals, np.zeros((resize_amount, 1)), axis=0
        )

    def _record(self, x_orig, x, fval_orig, fsd, fun_evaltime):
        duplicateFlag = False

        if duplicateFlag:
            # toDo: handling of duplicate entries
            pass
        else:
            self.Xn += 1
            if self.Xn > self.x_orig.shape[0] - 1:
                self._expand_arrays()

            # record function time
            if fun_evaltime is not None:
                self.fun_evaltime[self.Xn] = fun_evaltime
                self.total_fun_evaltime += fun_evaltime

            self.x_orig[self.Xn] = x_orig
            self.x[self.Xn] = x
            self.y_orig[self.Xn] = fval_orig
            fval = fval_orig
            if self.transform_parameters:
                fval += self.parameter_transformer.log_abs_det_jacobian(
                    np.reshape(x, (1, x.shape[0]))
                )[0]
            self.y[self.Xn] = fval
            if fsd is not None:
                self.S[self.Xn] = fsd
            self.X_flag[self.Xn] = True
            self.nevals[self.Xn] = max(1, self.nevals[self.Xn] + 1)
            self.ymax = np.amax(self.y[self.X_flag])
            return fval, self.Xn
