from bayesjax.conjugate.base import ConjugateModel
from typing import Union
from jax import numpy as jnp, random
from jax.typing import ArrayLike
from jax import Array, lax, vmap
from jax.scipy.special import betaln, gammaln, multigammaln
from jax.scipy.stats import t, nbinom

"""
Implements the following conjugate models (using the base ConjugateModel):
1. Beta - Bernoulli
2. Normal - Normal-Inverse Gamma (univariate)
3. Normal - Normal-Inverse Wishart (multivariate)
4. Categorical - Dirichlet
5. Poisson - Gamma
6. Gamma - Gamma (known rate), includes the exponential
7. Normal - Normal for location family Gaussian (univariate) with known scale sigma
"""

class BetaBinomial(ConjugateModel):
    """
    observations follow Binomial(n, theta)
    prior on theta is Beta(alpha, beta)
    prior_params = (alpha, beta)
    """
    def __init__(self, alpha: Union[float, Array], beta: Union[float, Array]):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ["theta"]

    def posterior_params(self, data: ArrayLike) -> "BetaBinomial":
        data = jnp.atleast_2d(data)
        x = data[:, 0]
        n = data[:, 1]
        successes = jnp.sum(x)
        failures = jnp.sum(n - x)
        return BetaBinomial(
            alpha=self.alpha + successes,
            beta=self.beta + failures
        )

    def posterior_from_stats(self, stats: ArrayLike) -> "BetaBinomial":
        # stats are successes and failures -> array of size 2
        return BetaBinomial(
            alpha=self.alpha + stats[0],
            beta=self.beta + stats[1]
        )

    def sample(self, rng_key, num_samples: int = 1) -> Array:
        return random.beta(rng_key, self.alpha, self.beta, shape=(num_samples,))

    def mean_(self):
        return {
            "theta": self.alpha / (self.alpha + self.beta)
        }

    def variance_(self):
        return {
            "theta": self.alpha * self.beta / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)
        }

    def log_marginal_likelihood(self, data: ArrayLike) -> Array:
        data = jnp.atleast_2d(data)
        x = data[:, 0]
        n = data[:, 1]
        log_binom_coeffs = jnp.sum(
            gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
        )
        successes = jnp.sum(x)
        failures = jnp.sum(n - x)
        return (
                log_binom_coeffs +
                betaln(self.alpha + successes, self.beta + failures) -
                betaln(self.alpha, self.beta)
        )

    def predictive_logpdf(self, x_new: Array, data: ArrayLike) -> Array:
        data = jnp.atleast_2d(data)
        x_new = jnp.atleast_2d(x_new)
        x = data[:, 0]
        n = data[:, 1]
        successes = jnp.sum(x)
        failures = jnp.sum(n - x)
        post_alpha = self.alpha + successes
        post_beta = self.beta + failures
        return betaln(post_alpha + x_new[:,0], post_beta + x_new[:,1] - x_new[:,0]) - betaln(post_alpha, post_beta)

class NormalNormalInvGamma(ConjugateModel):
    """
    observations follow N(mu, sigma2) univariate
    prior on (mu, sigma2) is Normal-Inverse Gamma
    prior_params = (mu0, kappa0, alpha0, beta0)
    note: beta0 is rate parameter
    """
    def __init__(
        self,
        mu0: ArrayLike,
        kappa0: ArrayLike,
        alpha0: ArrayLike,
        beta0: ArrayLike,
    ):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.param_names = ["mu", "sigma2"]

    def posterior_params(self, data: Array) -> "NormalNormalInvGamma":
        x = jnp.atleast_1d(data)
        n = x.size
        mean_x = jnp.mean(x)
        sum_sq_diff = jnp.sum((x - mean_x) ** 2)

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * mean_x) / kappa_n
        alpha_n = self.alpha0 + n / 2
        beta_n = self.beta0 + 0.5 * sum_sq_diff + \
                 (self.kappa0 * n * (mean_x - self.mu0) ** 2) / (2 * kappa_n)

        return NormalNormalInvGamma(mu_n, kappa_n, alpha_n, beta_n)

    def predictive_logpdf(self, x_new: ArrayLike, data: Array) -> Array:
        post = self.posterior_params(data)
        dof = 2 * post.alpha0
        loc = post.mu0
        scale = jnp.sqrt(post.beta0 * (1 + 1 / post.kappa0) / post.alpha0)
        return t.logpdf(x_new, df=dof, loc=loc, scale=scale)

    def posterior_from_stats(self, stats: Array) -> "NormalNormalInvGamma":
        # stats is [sample size, sum, sum of squares]
        n, sum_x, sum_x2 = stats[0], stats[1], stats[2]
        def no_data_case():
            return {
                "mu0": self.mu0,
                "alpha0": self.alpha0,
                "beta0": self.beta0,
                "kappa0": self.kappa0,
            }

        def update_case():
            sample_mean = sum_x / n
            sample_var = sum_x2 - (sum_x ** 2) / n
            kappa_post = self.kappa0 + n
            mu_post = (self.kappa0 * self.mu0 + sum_x) / kappa_post
            alpha_post = self.alpha0 + n / 2
            beta_post = self.beta0 + 0.5 * sample_var + 0.5 * self.kappa0 * n * (sample_mean - self.mu0) ** 2 / kappa_post
            return {
                "mu0": mu_post,
                "alpha0": alpha_post,
                "beta0": beta_post,
                "kappa0": kappa_post,
            }

        params = lax.cond(n == 0, no_data_case, update_case)
        return NormalNormalInvGamma(**params)

    def sample(self, rng_key: Array, num_samples: int = 1) -> tuple[Array, Array]:
        key1, key2 = random.split(rng_key)
        sigma2_samples = self.beta0 / random.gamma(key1, self.alpha0, shape=(num_samples,))
        mu_samples = random.normal(key2, shape=(num_samples,)) * jnp.sqrt(sigma2_samples / self.kappa0) + self.mu0
        return mu_samples, sigma2_samples

    def mean_(self):
        if self.alpha0 > 1:
            mean_sigma2 = self.beta0 / (self.alpha0 - 1)
        else:
            mean_sigma2 = jnp.nan
        return {
            "mu": self.mu0,
            "sigma2": mean_sigma2,
        }

    def variance_(self):
        if self.alpha0 > 1:
            var_mu = self.beta0 / (self.kappa0 * (self.alpha0 - 1))
        else:
            var_mu = jnp.nan
        if self.alpha0 > 2:
            var_sigma2 = self.beta0 ** 2/ ((self.alpha0 - 2) * (self.alpha0 - 1) ** 2)
        else:
            var_sigma2 = jnp.nan
        return {
            "mu": var_mu,
            "sigma2": var_sigma2
        }



    def log_marginal_likelihood(self, data: Array) -> Array:
        x = jnp.atleast_1d(data)
        n = x.size
        mean_x = jnp.mean(x)
        sum_sq_diff = jnp.sum((x - mean_x) ** 2)

        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + n / 2
        beta_n = self.beta0 + 0.5 * sum_sq_diff + \
                 (self.kappa0 * n * (mean_x - self.mu0) ** 2) / (2 * kappa_n)

        return (
            self.alpha0 * jnp.log(self.beta0)
            - alpha_n * jnp.log(beta_n)
            + gammaln(alpha_n)
            - gammaln(self.alpha0)
            + 0.5 * (jnp.log(self.kappa0) - jnp.log(kappa_n))
        )

class PoissonGamma(ConjugateModel):
    """
    observations follow Poisson(lambda)
    prior on lambda is Gamma
    prior_params = (alpha, beta)
    note: beta is rate parameter
    """
    def __init__(self, alpha: Union[float, Array], beta: Union[float, Array]):
        self.alpha = alpha
        self.beta = beta
        self.param_names = ["lambda"]

    def posterior_params(self, data: ArrayLike) -> "PoissonGamma":
        x = jnp.atleast_1d(data)
        n = x.size
        return PoissonGamma(alpha=self.alpha+x.sum(), beta=self.beta+n)

    def posterior_from_stats(self, stats: ArrayLike) -> "PoissonGamma":
        # stats is [sample size, sum]
        return PoissonGamma(alpha=self.alpha + stats[1], beta=self.beta + stats[0])

    def sample(self, rng_key, num_samples: int = 1) -> Array:
        return random.gamma(rng_key, self.alpha,  shape=(num_samples,)) / self.beta

    def mean_(self):
        return {
            "lambda": self.alpha / self.beta
        }

    def variance_(self):
        return {
            "lambda": self.alpha / self.beta ** 2
        }

    def log_marginal_likelihood(self, data: ArrayLike) -> Array:
        x = jnp.atleast_1d(data)
        n = x.size
        S = x.sum()
        alpha = self.alpha + x.sum()
        beta = self.beta + n
        # log(Π 1/x_i!) = -Σ log(x_i!)
        log_factorial_term = -jnp.sum(gammaln(data + 1))

        log_marginal_likelihood = (
                log_factorial_term
                + alpha * jnp.log(beta)
                - gammaln(alpha)
                + gammaln(alpha + S)
                - (alpha + S) * jnp.log(beta + n)
        )
        return log_marginal_likelihood

    def predictive_logpdf(self, x_new: int, data: ArrayLike) -> Array:
        x = jnp.atleast_1d(data)
        n = x.size
        alpha_post = self.alpha + x.sum()
        beta_post = self.beta + n
        return nbinom.logpmf(x_new, alpha_post, beta_post / (beta_post + 1))

class CategoricalDirichlet(ConjugateModel):
    """
    observations follow Categorical(theta) - dim(theta) = K # categories
    prior on theta is Dirichlet
    prior_params = alpha[K]
    Note: can also use with multinomial Dirichlet
    """
    def __init__(self, alpha: ArrayLike):
        self.alpha = alpha
        self.K = len(alpha)
        self.param_names = ["theta"]

    def posterior_params(self, data: Array) -> "CategoricalDirichlet":
        # data is categorical e.g. [1,3,4,2,1,1,0,0]
        counts = jnp.bincount(data, length=len(self.alpha))
        alpha_n = self.alpha + counts
        return CategoricalDirichlet(alpha_n)

    def posterior_from_stats(self, stats: Array) -> "CategoricalDirichlet":
        # stats is count - array of size K
        alpha_n = self.alpha + stats
        return CategoricalDirichlet(alpha_n)

    def sample(self, rng_key: Array, num_samples: int = 1) -> Array:
        theta_samples = random.dirichlet(rng_key, self.alpha, shape=(num_samples,))
        return theta_samples

    def mean_(self):
        return {
            "theta": self.alpha / self.alpha.sum()
        }

    def variance_(self):
        alpha_tilde = self.alpha / self.alpha.sum()
        alpha0 = self.alpha.sum()
        return {
            "theta": alpha_tilde * (1 - alpha_tilde) / (alpha0 + 1)
        }

    def predictive_logpdf(self, x_new: ArrayLike, data: Array) -> Array:
        alpha_n = self.alpha + data
        return jnp.log(alpha_n[jnp.int32(x_new)]) - jnp.log(alpha_n.sum())


    def log_marginal_likelihood(self, data: Array) -> Array:
        alpha_n = self.alpha + data
        log_marginal_likelihood = gammaln(self.alpha.sum()) - gammaln(alpha_n.sum()) + (gammaln(alpha_n) - gammaln(self.alpha)).sum()
        return log_marginal_likelihood

class GammaGamma(ConjugateModel):
    """
    observations follow Gamma(alpha, beta) with known shape alpha
    prior on beta (rate) is Gamma
    prior_params = (alpha0, beta0)
    Note: known shape alpha is fed in (default 1 - exponential)
    """
    def __init__(self, alpha0: ArrayLike, beta0: ArrayLike, alpha: ArrayLike):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha = alpha
        self.param_names = ["beta"]

    def posterior_params(self, data: Array) -> "GammaGamma":
        x = jnp.atleast_1d(data)
        n = x.size
        return GammaGamma(
            alpha0=self.alpha0 + n*self.alpha,
            beta0=self.beta0 + x.sum(),
            alpha=self.alpha
        )

    def posterior_from_stats(self, stats: Array) -> "GammaGamma":
        # stats is [sample size, sum]
        return GammaGamma(
            alpha0=self.alpha0 + stats[0] * self.alpha,
            beta0=self.beta0 + stats[1],
            alpha=self.alpha
        )

    def sample(self, rng_key: Array, num_samples: int = 1) -> Array:
        samples = random.gamma(rng_key, self.alpha0, shape=(num_samples,)) / self.beta0
        return samples

    def mean_(self):
        return {
            "beta": self.alpha0 / self.beta0
        }

    def variance_(self):
        return {
            "beta": self.alpha0 / self.beta0 ** 2
        }

    def log_marginal_likelihood(self, data: Array) -> Array:
        x = jnp.atleast_1d(data)
        n = x.size
        alpha, alpha0, beta0 = self.alpha, self.alpha0, self.beta0
        alphan = alpha0 + n * alpha
        betan = beta0 + x.sum()
        log_lik = (
                n * alpha * jnp.log(beta0)
                - n * gammaln(alpha)
                + gammaln(alphan)
                - gammaln(alpha0)
                - alphan * jnp.log(betan)
                + alpha0 * jnp.log(beta0)
                + (alpha - 1) * jnp.sum(jnp.log(data))
        )
        return log_lik

    def predictive_logpdf(self, x_new: ArrayLike, data: Array) -> Array:
        # Lomax (Pareto type 2) distribution
        x = jnp.atleast_1d(data)
        n = x.size
        alpha, alpha0, beta0 = self.alpha, self.alpha0, self.beta0
        alphan = alpha0 + n * alpha
        betan = beta0 + x.sum()
        # Ensure x > 0
        x_new = jnp.asarray(x_new)
        log_pdf = (
                gammaln(alphan + alpha)
                - gammaln(alphan)
                + alphan * jnp.log(betan)
                - (alphan + alpha) * jnp.log(x_new + betan)
                + (alpha - 1) * jnp.log(x_new)
                - gammaln(alpha)
        )
        return log_pdf

def sample_NIW_single(key, mu0, kappa0, nu0, psi0):
    """
        Sample (mu, Sigma) ~ NIW using Bartlett decomposition in pure JAX.
        Returns:
            mu: (num_samples, d)
            sigma: (num_samples, d, d)
        """
    d = mu0.shape[0]
    psi_inv = jnp.linalg.inv(psi0)

    keys = random.split(key, 3)
    k1, k2 = keys[0], keys[1]

    # Bartlett decomposition
    A = jnp.zeros((d, d))
    for j in range(d):
        A = A.at[j, j].set(jnp.sqrt(random.chisquare(k1, nu0 - j)))
        for k in range(j):
            A = A.at[j, k].set((j>k) * random.normal(k1))

    LA = A @ jnp.linalg.cholesky(psi_inv)
    W = LA @ LA.T
    Sigma = jnp.linalg.inv(W)

    # Sample mu ~ N(mu0, Sigma / kappa0)
    mu = random.multivariate_normal(keys[2], mu0, Sigma / kappa0)

    return mu, Sigma

def sample_NIW(key, mu0, kappa0, nu0, psi0, num_samples=1):
    keys = random.split(key, num_samples)
    samples = vmap(sample_NIW_single, in_axes=(0, None, None, None, None))(keys, mu0, kappa0, nu0, psi0)
    mu_samples, Sigma_samples = samples
    return mu_samples, Sigma_samples

def multivariate_t_logpdf(x: Array, loc: Array, scale: Array, df: Union[float, Array]) -> Array:
    """
    Log PDF of the multivariate Student-t distribution.

    Args:
        x: observation(s) (m,d)
        loc: mean vector (d,)
        scale: scale matrix (d, d)
        df: degrees of freedom > 0

    Returns:
        array of log density values
    """
    x = jnp.atleast_2d(x)  # Ensure x has shape (n, d)
    loc = jnp.atleast_1d(loc)
    d = x.shape[1]

    # Precompute constants
    inv_scale = jnp.linalg.inv(scale)
    logdet = jnp.linalg.slogdet(scale)[1]
    log_norm = (
            gammaln((df + d) / 2)
            - gammaln(df / 2)
            - 0.5 * (d * jnp.log(df * jnp.pi) + logdet)
    )

    def single_logpdf(xi):
        dev = xi - loc
        maha = dev @ inv_scale @ dev
        log_kernel = -0.5 * (df + d) * jnp.log1p(maha / df)
        return log_norm + log_kernel

    return vmap(single_logpdf)(x)

class NormalNormalInvWishart(ConjugateModel):
    """
    observations follow N(mu, Sigma) both unknown in dimension p>1
    prior on (mu,Sigma) is Normal-inverse Wishart
    prior_params = (mu0, kappa0, nu0, Psi0)
    Note: dimension from given prior - no dim check for data done
    """
    def __init__(self, mu0: ArrayLike, kappa0: Union[float, Array], psi0: ArrayLike, nu0: Union[float, Array]):
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.psi0 = psi0
        self.nu0 = nu0
        self.d = mu0.shape[0]

    def posterior_params(self, data: ArrayLike) -> "NormalNormalInvWishart":
        x = jnp.asarray(data)
        n = x.shape[0]
        x_bar = jnp.mean(x, axis=0)
        S = jnp.cov(x, rowvar=False, bias=True) * n  # scatter matrix

        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n
        mu_n = (self.kappa0 * self.mu0 + n * x_bar) / kappa_n
        delta = (x_bar - self.mu0)[..., None]
        psi_n = self.psi0 + S + (self.kappa0 * n / kappa_n) * (delta @ delta.T)

        return NormalNormalInvWishart(mu_n, kappa_n, psi_n, nu_n)

    def posterior_from_stats(self, stats: tuple[int, Array, Array]) -> "NormalNormalInvWishart":
        n, x_bar, S = stats
        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n
        mu_n = (self.kappa0 * self.mu0 + n * x_bar) / kappa_n
        delta = (x_bar - self.mu0)[..., None]
        psi_n = self.psi0 + S + (self.kappa0 * n / kappa_n) * (delta @ delta.T)
        return NormalNormalInvWishart(mu_n, kappa_n, psi_n, nu_n)

    def sample(self, rng_key: Array, num_samples: int) -> tuple[Array, Array]:
        mu_samples, Sigma_samples = sample_NIW(rng_key, self.mu0, self.kappa0, self.nu0, self.psi0, num_samples)
        return mu_samples, Sigma_samples

    def mean_(self) -> dict:
        return {
            "mu": self.mu0,
            "Sigma": self.psi0 / (self.nu0 - self.d - 1)
        }

    def variance_(self) -> dict:
        # Only defined for marginal components of mu
        var_mu = self.psi0 / (self.kappa0 * (self.nu0 - self.d - 1))
        return {"mu": jnp.diag(var_mu)}

    def log_marginal_likelihood(self, data: ArrayLike) -> float:
        x = jnp.asarray(data)
        n, d = x.shape
        x_bar = jnp.mean(x, axis=0)
        S = jnp.cov(x, rowvar=False, bias=True) * n

        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n
        delta = (x_bar - self.mu0)[..., None]
        psi_n = self.psi0 + S + (self.kappa0 * n / kappa_n) * (delta @ delta.T)

        logZ0 = (
                -n * d / 2 * jnp.log(jnp.pi)
                + multigammaln(self.nu0 / 2, d)
                + self.nu0 / 2 * jnp.linalg.slogdet(self.psi0)[1]
                - d / 2 * jnp.log(self.kappa0)
        )
        logZn = (
                multigammaln(nu_n / 2, d)
                + (nu_n / 2) * jnp.linalg.slogdet(psi_n)[1]
                - d / 2 * jnp.log(kappa_n)
        )

        return logZ0 - logZn

    def predictive_logpdf(self, x_new: ArrayLike, data: ArrayLike) -> Array:
        post = self.posterior_params(data)
        mu = post.mu0
        scale = post.psi0 * (post.kappa0 + 1) / (post.kappa0 * (post.nu0 - post.d + 1))
        df = post.nu0 - post.d + 1
        return multivariate_t_logpdf(x=x_new, loc=mu, scale=scale, df=df)