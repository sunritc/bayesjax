# models/conjugate/interfaces.py
from typing import Protocol, Any
from jax.typing import ArrayLike
from jax import Array

class ConjugateModelProtocol(Protocol):
    """base class for defining conjugate models"""
    def sample_posterior(self, rng_key, data: ArrayLike, num_samples: int = 1) -> Array: ...
    def marginal_likelihood(self, data: ArrayLike) -> float: ...
    def predictive_distribution(self, x_new: Any, data: ArrayLike) -> Any: ...