# models/conjugate/interfaces.py
from typing import Protocol, Any
from abc import ABC, abstractmethod
from jax.typing import ArrayLike
from jax import Array


class ConjugateModel(ABC):
    # check param_names for dict keys
    """base class for defining conjugate models"""
    @abstractmethod
    def posterior_params(self, data: ArrayLike): ...
    """
    returns an object of the same class with updated posterior parameters from whole data
    """

    @abstractmethod
    def sample(self,  rng_key: Array, num_samples: int) -> ArrayLike: ...
    """
    samples num_samples samples from the object (seen as prior)
    """

    @abstractmethod
    def mean_(self) -> dict: ...
    """
    returns mean of each parameter in param_names
    """

    @abstractmethod
    def variance_(self) -> dict: ...
    """
    returns variance of each parameter (marginally) in param_names
    """

    @abstractmethod
    def log_marginal_likelihood(self, data: ArrayLike) -> float: ...
    """
    computes the marginal log likelihood of a given data under object prior
    """

    @abstractmethod
    def predictive_logpdf(self, x_new: Any, data: ArrayLike) -> Any: ...
    """
    computes the predictive distribution of a single new sample x_new under object prior given data
    """

    @abstractmethod
    def posterior_from_stats(self, stats: ArrayLike) : ...
    """
    returns an object of the same class with updated posterior parameters from sufficient statistics
    """