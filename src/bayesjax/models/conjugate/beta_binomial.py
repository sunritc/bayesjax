# models/conjugate/beta_binomial.py

from dataclasses import dataclass
from typing import Union
from jax import numpy as jnp, random
from jax.typing import ArrayLike
from jax import Array
from jax.scipy.special import betaln, gammaln

@dataclass
class BetaBinomial:
    alpha: Union[float, Array]
    beta: Union[float, Array]

    def posterior_params(self, data: ArrayLike) -> "BetaBinomial":
        """
        Return BetaBinomial class with posterior updated parameters.
        :param data
        """
        data = jnp.atleast_2d(data)
        x = data[:, 0]
        n = data[:, 1]
        successes = jnp.sum(x)
        failures = jnp.sum(n - x)
        return BetaBinomial(
            alpha=self.alpha + successes,
            beta=self.beta + failures
        )

    def sample_posterior(self, rng_key, data: ArrayLike, num_samples: int = 1) -> Array:
        """
        Sample from the posterior Beta distribution given binomial observations.
        Each row in data is (x_i, n_i): successes and trials.
        """
        post = self.posterior_params(data)
        return random.beta(rng_key, post.alpha, post.beta, shape=(num_samples,))

    def posterior_sample_fn(self, data: ArrayLike):
        """
        Returns a sampler object that can be used to sample from the posterior.
        :param data:
        :return: sampler object
        """
        def sampler(rng_key: Array, num_samples: int = 1) -> Array:
            post = self.posterior_params(data)
            return random.beta(rng_key, post.alpha, post.beta, shape=(num_samples,))

        return sampler

    def marginal_likelihood(self, data: ArrayLike) -> Array:
        """
        Compute log marginal likelihood for Binomial observations under a Beta prior.
        Uses closed-form expression:
            log p(data | alpha, beta) = sum_i log C(n_i, x_i)
                + betaln(alpha + sum x_i, beta + sum (n_i - x_i)) - betaln(alpha, beta)
        """
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

    def predictive_distribution(self, x_new: int, n_new: int, data: ArrayLike) -> Array:
        """
        Predictive probability of x_new successes in n_new trials using the Beta-Binomial predictive distribution.
        """
        data = jnp.atleast_2d(data)
        x = data[:, 0]
        n = data[:, 1]
        successes = jnp.sum(x)
        failures = jnp.sum(n - x)
        post_alpha = self.alpha + successes
        post_beta = self.beta + failures
        return jnp.exp(
            betaln(post_alpha + x_new, post_beta + n_new - x_new)
            - betaln(post_alpha, post_beta)
        )