# R̂, ESS, etc.
from typing import Union
import jax.numpy as jnp
import jax

def compute_rhat(chains: jnp.ndarray) -> Union[float, jax.Array]:
    """Compute the potential scale reduction factor (R-hat).
    Assumes chains shape: (num_chains, num_samples, ...)
    """
    if chains.ndim < 2:
        raise ValueError("Chains array must have at least 2 dimensions (num_chains, num_samples)")

    num_chains, num_samples = chains.shape[:2]

    chain_means = jnp.mean(chains, axis=1)
    chain_vars = jnp.var(chains, axis=1, ddof=1)

    between_chain_var = jnp.var(chain_means, axis=0, ddof=1) * num_samples
    within_chain_var = jnp.mean(chain_vars, axis=0)

    var_hat = ((num_samples - 1) / num_samples) * within_chain_var + (1 / num_samples) * between_chain_var
    rhat = jnp.sqrt(var_hat / within_chain_var)
    return rhat

def compute_ess(chains: jnp.ndarray) -> Union[float, jax.Array]:
    """Compute the effective sample size (ESS).
    Assumes chains shape: (num_chains, num_samples, ...)
    """
    if chains.ndim < 2:
        raise ValueError("Chains array must have at least 2 dimensions (num_chains, num_samples)")

    num_chains, num_samples = chains.shape[:2]

    # Reshape to (num_chains * num_samples, ...)
    reshaped = chains.reshape((-1,) + chains.shape[2:])

    # Compute variance across all samples
    var_all = jnp.var(reshaped, axis=0, ddof=1)

    # Compute variance within each chain
    var_within = jnp.var(chains, axis=1, ddof=1)
    mean_within = jnp.mean(var_within, axis=0)

    ess = (num_chains * num_samples) * var_all / mean_within
    return ess
