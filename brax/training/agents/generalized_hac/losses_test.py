from absl.testing import parameterized
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from brax.training.agents.generalized_hac import losses as ghac_losses

class GeneralizedHorizonLossesTest(parameterized.TestCase):
    def testComputeTDValues(self):
        truncation = jnp.array([0., 0., 0., 0., 0.])
        termination = jnp.array([0., 0., 0., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([10., 10., 10., 10., 10.])
        bootstrap_value = jnp.array(100_000.)

        td_value = ghac_losses.compute_generalized_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            actor_xi=0.9,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(67969))

    def testComputeTDValuesTerminated(self):
        truncation = jnp.array([0., 0., 0., 0., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = ghac_losses.compute_generalized_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            actor_xi=0.9,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(2_359.))

    def testComputeTDValuesTruncated(self):
        truncation = jnp.array([0., 0., 0., 1., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., 4_000., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = ghac_losses.compute_generalized_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            actor_xi=0.9,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(3_088.))

    def testComputeTDValuesReset(self):
        truncation = jnp.array([0., 0., 1., 0., 0.])
        termination = jnp.array([0., 0., 1., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., 4_000., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = ghac_losses.compute_generalized_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            actor_xi=0.9,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(3_250.))

if __name__ == '__main__':
    absltest.main()
