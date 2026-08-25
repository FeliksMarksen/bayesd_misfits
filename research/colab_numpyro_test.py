"""Quick NumPyro x JAX compatibility test on Colab."""
from __future__ import annotations

import sys

print("Python:", sys.version)

try:
    import jax
    print("jax:", jax.__version__)
    print("devices:", jax.devices())
except Exception as e:
    print(f"JAX import failed: {e}")
    sys.exit(1)

try:
    import numpyro
    print("numpyro:", numpyro.__version__)
    from numpyro.infer import MCMC, NUTS
    print("✓ NumPyro MCMC/NUTS import OK")
except Exception as e:
    print(f"NumPyro import failed: {e}")
    sys.exit(1)

try:
    import jax.numpy as jnp
    import numpyro.distributions as dist

    def model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(x, 1), obs=jnp.array([1.0, 2.0, 3.0]))

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    print("✓ Tiny MCMC run completed")
    print("x mean:", float(mcmc.get_samples()["x"].mean()))
except Exception as e:
    print(f"MCMC run failed: {e}")
    sys.exit(1)

print("\n✓ NumPyro x JAX compatibility test passed.")
