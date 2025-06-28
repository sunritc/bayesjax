from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import t
from jax.scipy.special import gammaln
from jax import Array

ArrayLike = Union[float, Array]

@dataclass
class NormalInverseGamma:
    mu0: ArrayLike
    kappa0: ArrayLike
    alpha0: ArrayLike
    beta0: ArrayLike

    def posterior_params(self, data: Array) -> "NormalInverseGamma":
        x = jnp.atleast_1d(data)
        n = x.size
        mean_x = jnp.mean(x)
        sum_sq_diff = jnp.sum((x - mean_x) ** 2)

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * mean_x) / kappa_n
        alpha_n = self.alpha0 + n / 2
        beta_n = self.beta0 + 0.5 * sum_sq_diff + \
                 (self.kappa0 * n * (mean_x - self.mu0) ** 2) / (2 * kappa_n)

        return NormalInverseGamma(mu_n, kappa_n, alpha_n, beta_n)

    def sample_posterior(self, rng_key: Array, data: Array, num_samples: int = 1) -> tuple[Array, Array]:
        post = self.posterior_params(data)
        key1, key2 = random.split(rng_key)
        sigma2_samples = random.gamma(key1, post.alpha0, shape=(num_samples,)) ** -1 * post.beta0
        mu_samples = random.normal(key2, shape=(num_samples,)) * jnp.sqrt(sigma2_samples / post.kappa0) + post.mu0
        return mu_samples, sigma2_samples

    def predictive_distribution(self, x_new: ArrayLike, data: Array) -> Array:
        post = self.posterior_params(data)
        dof = 2 * post.alpha0
        loc = post.mu0
        scale = jnp.sqrt(post.beta0 * (1 + 1 / post.kappa0) / post.alpha0)
        return t.pdf(x_new, df=dof, loc=loc, scale=scale)

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