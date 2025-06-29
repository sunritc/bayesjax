# bayes_mcmc/models/mixture_models/normal_univariate.py

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
from jax import random
from bayesjax.core.base import MCMCModel
from bayesjax.models.conjugate.normal_univariate import NormalNormalInvGamma
from bayesjax.models.conjugate.categorical_dirichlet import CategoricalDirichlet


def get_mixture_model(kind: str, **kwargs) -> MCMCModel:
    if kind == "full":
        return NormalMixtureFullGibbs(**kwargs)
    elif kind == "collapsed":
        raise ValueError(f"Not yet implemented")
    else:
        raise ValueError(f"Unknown kind: {kind}")


class NormalMixtureFullGibbs(MCMCModel):
    """
    Initialize a mixture model with a conjugate Normal-Inverse-Gamma prior.

    Args:
        num_components: Number of mixture components (K)
        prior: tuple (Prior over (mu, sigma²) using Normal-Inverse-Gamma, Prior over pi using Categorical-Dirichlet)
    """
    def __init__(self, num_components: int, prior: (NormalNormalInvGamma, CategoricalDirichlet)):
        self.K = num_components
        self.prior = prior[0]
        self.prior_latent = prior[1]

    def initialize(self, key: jax.Array, data: jnp.ndarray) -> Dict[str, Any]:
        """
        Randomly initialize the mixture state.

        Args:
            key: PRNGKey for randomness
            data: Observed data, shape (N,)

        Returns:
            Initialized MixtureState

        Note: state is a dict with keys 'z' and 'params' (mus, sigmas)
        Note: sigma is used, NOT sigma^2
        """
        N = data.shape[0]
        key_z, key_params = random.split(key)
        z = random.randint(key_z, shape=(N,), minval=0, maxval=self.K)
        mus = random.normal(key_params, shape=(self.K,))
        sigmas = jnp.ones((self.K,))
        weights = jnp.ones((self.K,)) / self.K

        return {'z': z, 'params': (mus, sigmas, weights)}

    def _sample_all_components(self, key: jax.Array, data: jnp.ndarray, z: jnp.ndarray) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Precompute sufficient statistics per component
        K = self.K
        counts = jnp.bincount(z, length=K)
        sum_x = jnp.bincount(z, weights=data, length=K)
        sum_x2 = jnp.bincount(z, weights=data ** 2, length=K)

        # update mu, sigma from NIG model
        def single_component(i, key):
            n_k = counts[i]
            sx = sum_x[i]
            sx2 = sum_x2[i]

            post = self.prior.posterior_from_stats(n_k, sx, sx2)
            mu, sigma2 = post.sample(key)
            return mu, jnp.sqrt(sigma2)  # (mu, sigma)

        keys = random.split(key, K+1)
        samples = jax.vmap(single_component)(jnp.arange(K), keys[:K])
        mus = jnp.squeeze(samples[0], axis=-1)
        sigmas = jnp.squeeze(samples[1], axis=-1)

        # update weights from Categorical Dirichlet model
        post = self.prior_latent.posterior_params(counts)
        weights = post.sample(keys[-1])[0]
        return mus, sigmas, weights

    def step(self, key: jax.Array, data: jnp.ndarray, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single Gibbs sampling step.

        Args:
            key: PRNGKey for this iteration
            data: Observed data, shape (N,)
            state: Current sampler state

        Returns:
            Updated sampler state
        """
        mus, sigmas, weights = state['params']
        N = data.shape[0]

        key_z, key_params, key_assign = random.split(key, 3)

        # Sample component assignments
        def log_prob(x, mu, sigma):
            return -0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((x - mu) ** 2) / (sigma**2) # re

        def sample_z(x, key):
            logps = jnp.array([log_prob(x, mus[k], sigmas[k]) for k in range(self.K)])
            sample =  random.categorical(key, logps + jnp.log(weights))
            return sample

        keys = random.split(key_assign, N)
        new_z = jax.vmap(sample_z, in_axes=(0, 0))(data, keys)
        # Sample new component parameters
        new_params = self._sample_all_components(key_params, data, new_z)

        return {'z': new_z, 'params': new_params}

    def extract_sample(self, state: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        return {
            'z': state['z'],
            'mu': state['params'][0],
            'sigma': state['params'][1],
            'weights': state['params'][2]
        }
