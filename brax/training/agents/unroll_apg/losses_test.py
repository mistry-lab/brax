from absl.testing import parameterized
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from brax.training.agents.unroll_apg import losses as apg_losses

class DiffRLLossesTest(parameterized.TestCase):
    def testComputeDiscountedReward(self):
        truncation = jnp.array([0., 0., 0., 0., 0.])
        termination = jnp.array([0., 0., 0., 0., 0.])
        rewards = jnp.array([-1., -10., -200., -3_000., 10_000.])

        td_value = apg_losses.compute_discounted_reward(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(4_202.))

    def testComputeDiscountedRewardTerminated(self):
        truncation = jnp.array([0., 0., 0., 0., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([-1., -10., -200., -3_000., 10_000.])

        td_value = apg_losses.compute_discounted_reward(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(-2_359.))

    def testComputeDiscountedRewardTruncated(self):
        truncation = jnp.array([0., 0., 0., 1., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([-1., -10., -200., -3_000., 10_000.])

        td_value = apg_losses.compute_discounted_reward(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(-172.))

    def testComputeDiscountedRewardReset(self):
        truncation = jnp.array([0., 0., 1., 0., 0.])
        termination = jnp.array([0., 0., 1., 0., 0.])
        rewards = jnp.array([-1., -10., -200., -3_000., 10_000.])

        td_value = apg_losses.compute_discounted_reward(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(-10.))

if __name__ == '__main__':
    absltest.main()
