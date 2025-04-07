import jax
import jax.numpy as jnp
from brax.training.distribution import NormalDistribution

def sample(parameters, prng):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    return NormalDistribution(loc=loc, scale=scale).sample(prng)[0]

def main():
    params = jnp.asarray([2.0, 1.0])
    grad = jax.grad(sample)(params, jax.random.PRNGKey(1))
    print(grad)

if __name__ == '__main__':
    main()
