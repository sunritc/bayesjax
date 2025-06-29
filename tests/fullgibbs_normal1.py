import jax
import jax.numpy as jnp
from bayesjax.models.mixture_models.normal_univariate import NormalMixtureFullGibbs
from bayesjax.models.conjugate.normal_univariate import NormalNormalInvGamma
from bayesjax.models.conjugate.categorical_dirichlet import CategoricalDirichlet
from bayesjax.core.diagnostics import compute_ess
from jax import debug

NUM_CHAINS = 4

def generate_synthetic_data(key, N=1000):
    key1, key2 = jax.random.split(key)
    p = N // 3
    data1 = jax.random.normal(key1, (p,)) * 1.0 + 0.0  # Cluster 1
    data2 = jax.random.normal(key2, (N-p,)) * 0.5 + 6.0  # Cluster 2
    data = jnp.concatenate([data1, data2])
    return data

def test_gibbs_mixture():
    key = jax.random.PRNGKey(0)
    data = generate_synthetic_data(key)

    # Define prior
    prior = NormalNormalInvGamma(mu0=0.0, kappa0=2.0, alpha0=5.0, beta0=1.0)
    prior_latent = CategoricalDirichlet(alpha0=jnp.ones(2))

    # Define model
    model = NormalMixtureFullGibbs(num_components=2, prior=(prior, prior_latent))

    # Run MCMC
    keys = jax.random.split(key, NUM_CHAINS)
    samples = model.run_multiple_chains(
        keys=keys,
        data=data,
        num_iters=10000,
        burnin_proportion=0.5,
        thinning=10
    )

    # Diagnostics per component
    print(samples["mu"][:,-1])
    print(samples["sigma"][:,-1])
    print(samples["weights"][:,-1])

if __name__ == "__main__":
    test_gibbs_mixture()