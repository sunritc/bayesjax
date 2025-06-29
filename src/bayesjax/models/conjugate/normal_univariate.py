from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import random, lax, Array
from jax.scipy.stats import t
from jax.scipy.special import gammaln

ArrayLike = Union[float, Array]

@dataclass
class NormalNormalInvGamma:
    mu0: ArrayLike
    kappa0: ArrayLike
    alpha0: ArrayLike
    beta0: ArrayLike

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

    def sample(self, rng_key: Array, num_samples: int = 1) -> tuple[Array, Array]:
        key1, key2 = random.split(rng_key)
        sigma2_samples = self.beta0 / random.gamma(key1, self.alpha0, shape=(num_samples,))
        mu_samples = random.normal(key2, shape=(num_samples,)) * jnp.sqrt(sigma2_samples / self.kappa0) + self.mu0
        return mu_samples, sigma2_samples

    def predictive_distribution(self, x_new: ArrayLike, data: Array) -> Array:
        post = self.posterior_params(data)
        dof = 2 * post.alpha0
        loc = post.mu0
        scale = jnp.sqrt(post.beta0 * (1 + 1 / post.kappa0) / post.alpha0)
        return t.pdf(x_new, df=dof, loc=loc, scale=scale)

    def posterior_from_stats(self, n: Array, sum_x: Array, sum_x2: Array) -> "NormalNormalInvGamma":
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