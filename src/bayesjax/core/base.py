from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, lax

from bayesjax.core.diagnostics import compute_rhat, compute_ess

Array = jax.Array
MCMCSamples = Dict[str, Array]
Diagnostics = Dict[str, Dict[str, Union[float, Array]]]

from jax import debug

class MCMCModel(ABC):
    @abstractmethod
    def initialize(self, key: Array, data: Array) -> Dict:
        ...

    @abstractmethod
    def step(self, key: Array, data: Array, state: Dict) -> Dict:
        ...

    @abstractmethod
    def extract_sample(self, state: Dict) -> MCMCSamples:
        ...

    def run_mcmc(
        self,
        key: Array,
        data: Array,
        num_iters: int,
        burnin_proportion: float = 0.1,
        thinning: int = 1,
    ) -> MCMCSamples:
        burnin = int(burnin_proportion * num_iters)
        keep_idxs = jnp.arange(num_iters)
        keep_mask = (keep_idxs >= burnin) & ((keep_idxs - burnin) % thinning == 0)

        def scan_step(state, key):
            new_state = self.step(key, data, state)
            sample = self.extract_sample(new_state)
            return new_state, sample

        key_init, key_scan = random.split(key)
        state = self.initialize(key_init, data)
        keys = random.split(key_scan, num_iters)

        _, raw_samples = lax.scan(scan_step, state, keys)

        # Apply filter to retain only post-burnin, thinned samples
        # jnp.nonzero(keep_mask, )
        # def apply_filter(x):
        #     return x[keep_mask]

        return raw_samples #.tree.map(apply_filter, raw_samples)

    def run_multiple_chains(
        self,
        keys: Array,
        data: Array,
        num_iters: int,
        burnin_proportion: float = 0.1,
        thinning: int = 1,
    ) -> Tuple[MCMCSamples, Diagnostics]:
        run_chain_jit = jax.jit(partial(
            self.run_mcmc,
            data=data,
            num_iters=num_iters,
            burnin_proportion=burnin_proportion,
            thinning=thinning
        ))

        chains = jax.vmap(run_chain_jit)(keys)  # dict[param] -> (nchains, nsamples, ...)

        # rhat = compute_rhat(chains)
        # ess = compute_ess(chains)
        #
        # diagnostics = {
        #     param: {
        #         "rhat": rhat[param],
        #         "ess": ess[param]
        #     }
        #     for param in chains
        # }


        return chains