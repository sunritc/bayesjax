from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import random, lax, Array
from jax.scipy.stats import t
from jax.scipy.special import gammaln

ArrayLike = Union[float, Array]

@dataclass
class CategoricalDirichlet:

    def __init__(self, alpha0: ArrayLike):
        self.alpha0 = alpha0
        self.K = len(alpha0)

    def posterior_params(self, data: Array) -> "CategoricalDirichlet":
        # data is counts
        alpha_n = self.alpha0 + data
        return CategoricalDirichlet(alpha_n)

    def sample(self, rng_key: Array, num_samples: int = 1) -> Array:
        theta_samples = random.dirichlet(rng_key, self.alpha0, shape=(num_samples,))
        return theta_samples

    def predictive_distribution(self, x_new: ArrayLike, data: Array) -> Array:
        alpha_n = self.alpha0 + data
        return alpha_n[jnp.int32(x_new)] / alpha_n.sum()


    def log_marginal_likelihood(self, data: Array) -> Array:
        alpha_n = self.alpha0 + data
        log_marginal_likelihood = gammaln(self.alpha0.sum()) - gammaln(alpha_n.sum()) + (gammaln(alpha_n) - gammaln(self.alpha0)).sum()
        return log_marginal_likelihood