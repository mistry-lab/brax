from absl.testing import parameterized
from absl.testing import absltest

from brax.envs.fd.linear import Linear
from brax.training.agents.diffrl_shac.unroll import generate_batched_unroll

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

def policy(x: float, key_sample: jax.random.PRNGKey):
    return jnp.ones_like(x), {}

class DiffRLLossesTest(parameterized.TestCase):
    def testUnroll(self):
        env = Linear()
        rng = jax.random.PRNGKey(seed=0)
        key_env, key_unroll = jax.random.split(rng)

        env_keys = jax.random.split(key_env, 2)
        env_state = jax.vmap(env.reset)(env_keys)

        final_state, data = generate_batched_unroll(
            env=env,
            env_state=env_state,
            policy=policy,
            key=key_unroll,
            reward_scaling=1.0,
            unroll_length=3,
            number=5,
        )

        self.assertEqual(data.observation.shape, (10, 3))

if __name__ == '__main__':
    absltest.main()
