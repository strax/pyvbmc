import numpy as np


class VariationalPosterior(object):
    """
    Variational Posterior class
    """

    def __init__(self):
        self.d = None  # number of dimensions
        self.k = None  # number of components
        self.w = None
        self.mu = None
        self.sigma = None
        self.lamb = None
        self.parameter_transformer = None
        self.optimize_mu = None
        self.optimize_sigma = None
        self.optimize_lamb = None
        self.optimize_weights = None
        self.bounds = None

    def sample(
        self,
        n: int,
        origflag: bool = True,
        balanceflag: bool = False,
        df: float = np.inf,
    ):
        """
        sample random samples from variational posterior

        Parameters
        ----------
        N : int
            number of samples
        origflag : bool, optional
            returns the random vectors in the original
            parameter space if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True
        balanceflag : bool, optional
            balanceflag=True balances the generating process
            such that the random samples in X come from each
            mixture component exactly proportionally
            (or as close as possible) to the variational mixture weights.
            If balanceflag=False (default), the generating mixture
            for each sample is determined randomly according to
            the mixture weights, by default False
        df : float, optional
            samples generated from a heavy-tailed version
            of the variational posterior, in which
            the multivariate normal components have been replaced by
            multivariate t-distributions with DF degrees of freedom.
            The default is df=Inf, limit in which the t-distribution
            becomes a multivariate normal., by default np.inf

        Returns
        -------
        (X : np.ndarray, I : np.ndarray)
            X : N-by-D matrix X of random vectors chosen
            from the variational posterior
            I : N-by-1 array such that the n-th
                element of I indicates the index of the
                variational mixture component from which
                the n-th row of X has been generated.
        """
        # missing to sample from gp
        gp_sample = False
        if n < 1:
            x = np.zeros((0, self.d))
            i = np.zeros((0, 1))
            return x, i
        elif gp_sample:
            pass
        else:
            rng = np.random.default_rng()
            if self.k > 1:
                if balanceflag:
                    # exact split of samples according to mixture weigths
                    repeats = np.floor(self.w * n).astype("int")
                    i = np.repeat(range(self.k), repeats.flatten())

                    # compute remainder samples (with correct weights) if needed
                    if n > i.shape[0]:
                        w_extra = self.w * n - repeats
                        repeats_extra = np.ceil(np.sum(w_extra))
                        w_extra += self.w * (repeats_extra - sum(w_extra))
                        w_extra /= np.sum(w_extra)
                        i_extra = rng.choice(
                            range(self.k),
                            size=repeats_extra.astype("int"),
                            p=w_extra.flatten(),
                        )
                        i = np.append(i, i_extra)

                    rng.shuffle(i)
                    i = i[0:n]
                else:
                    i = rng.choice(range(self.k), size=n, p=self.w.flatten())

                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.conj().T[i]
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T[i]
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (n, 1)))
                    x = (
                        self.mu.conj().T[i]
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * t
                        * self.sigma.conj().T[i]
                    )
            else:
                if not np.isfinite(df) or df == 0:
                    x = (
                        self.mu.conj().T
                        + self.lamb.conj().T
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T
                    )
                else:
                    t = df / 2 / np.sqrt(rng.gamma(df / 2, df / 2, (n, 1)))
                    x = (
                        self.mu.conj().T
                        + self.lamb.conj().T
                        * t
                        * np.random.randn(n, self.d)
                        * self.sigma.conj().T
                    )
                i = np.zeros(n)
            if origflag:
                x = self.parameter_transformer.inverse(x)
        return x, i

    def pdf(
        self,
        x: np.ndarray,
        origflag: bool = True,
        logflag: bool = False,
        transflag: bool = False,
        df: float = np.inf,
    ):
        """
        pdf probability density function of VBMC posterior approximation

        Parameters
        ----------
        x : np.ndarray
            matrix of rows to evaluate the pdf at
            Rows of the N-by-D matrix x
            correspond to observations or points,
            and columns correspond to variables or coordinates.
        origflag : bool, optional
            returns the random vectors in the original
            parameter space if origflag=True (default),
            or in the transformed VBMC space
            if origflag=False, by default True
        logflag : bool, optional
            returns the value of the log pdf
            if LOGFLAG=True, otherwise
            the posterior density, by default False
        transflag : bool, optional
            transflag=True assumes
            that X is already specified in transformed VBMC space.
            Otherwise, X is specified
            in the original parameter space, by default False
        df : float, optional
            pdf of a heavy-tailed version
            of the variational posterior, in which
            the multivariate normal components have been replaced by
            multivariate t-distributions with DF degrees of freedom.
            The default is df=Inf, limit in which the t-distribution
            becomes a multivariate normal., by default np.inf

        Returns
        -------
        nd.array
            probability density of the variational posterior
            evaluated at each row of x.
        """
        # Convert points to transformed space
        if origflag and not transflag:
            x = self.parameter_tranformer(x)

        n, d = x.shape
        y = np.zeros(n, 1)
        pass
        return y

    def vbmc_moments(self, origflag, Ns):
        """
        VBMC_MOMENTS(VP) computes the mean MU and covariance
        """
        pass

    def vbmc_mode(self, nmax, origflag):
        """
        Find mode of VBMC posterior approximation.
        """
        pass

    def vbmc_mtv(self, vp2, Ns):
        """
        Marginal Total Variation distances between two variational posteriors.
        """
        pass

    def vbmc_power(self, n, cutoff):
        """
        Compute power posterior of variational approximation.
        """
        pass

    def vbmc_plot(self, vp_array, stats):
        """
        docstring
        """
        pass

    # private methods of vp class

    def __get_vptheta(
        self, optimize_mu, optimize_sigma, optimize_lambda, optimize_weights
    ):
        """
        Get vector of variational parameters from variational posterior.
        """
        pass

    def __robustSampleFromVP(self, Ns, Xrnd, quantile_thresh):
        """
        Robust sample from variational posterior
        """
        pass

    def __vbmc_output(self, parameter_list):
        """
        create output struct -> essentially a print?
        """
        pass

    def __vbmc_iterplot(self, parameter_list):
        """
        plot vp
        """
        pass