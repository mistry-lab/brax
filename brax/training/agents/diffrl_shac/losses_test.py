from absl.testing import parameterized
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from brax.training.agents.diffrl_shac import losses as shac_losses
from brax.training.gae import compute_vs

class DiffRLLossesTest(parameterized.TestCase):
    def testComputeGAE(self):
        truncation = jnp.array([0., 0., 0., 0.])
        termination = jnp.array([0., 0., 0., 0.])
        rewards = jnp.array([10., 200., 3_000., 10_000.])
        values = jnp.array([30_000., 30_000., 90_000., 90_000.])
        bootstrap_value = jnp.array(100_000.)

        vs = compute_vs(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.5,
            discount=0.8,
        )

        np.testing.assert_array_equal(vs, jnp.array([38_490., 66_200., 75_000., 90_000.]))

    def testComputeGAETerminated(self):
        truncation = jnp.array([0., 0., 0., 1.])
        termination = jnp.array([0., 0., 1., 1.])
        rewards = jnp.array([10., 200., 3_000., 10_000.])
        values = jnp.array([30_000., 30_000., 90_000., 90_000.])
        bootstrap_value = jnp.array(100_000.)

        vs = compute_vs(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.5,
            discount=0.8,
        )

        np.testing.assert_array_equal(vs, jnp.array([26_970., 37_400., 3_000., 90_000.]))

    def testComputeGAETruncated(self):
        truncation = jnp.array([0., 0., 1., 1.])
        termination = jnp.array([0., 0., 1., 1.])
        rewards = jnp.array([10., 200., 3_000., 10_000.])
        values = jnp.array([30_000., 30_000., 90_000., 90_000.])
        bootstrap_value = jnp.array(100_000.)

        vs = compute_vs(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.5,
            discount=0.8,
        )

        np.testing.assert_array_equal(vs, jnp.array([40_890., 72_200., 90_000., 90_000.]))

    def testComputeGAEReset(self):
        truncation = jnp.array([0., 0., 1., 0.])
        termination = jnp.array([0., 0., 1., 0.])
        rewards = jnp.array([10., 200., 3_000., 10_000.])
        values = jnp.array([30_000., 30_000., 90_000., 90_000.])
        bootstrap_value = jnp.array(10.)

        vs = compute_vs(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            lambda_=0.5,
            discount=0.8,
        )

        np.testing.assert_array_equal(vs, jnp.array([40_890., 72_200., 90_000., 10_008.]))

    def testComputeTDValues(self):
        truncation = jnp.array([0., 0., 0., 0., 0.])
        termination = jnp.array([0., 0., 0., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(67969))

    def testComputeTDValuesTerminated(self):
        truncation = jnp.array([0., 0., 0., 0., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(2_359.))

    def testComputeTDValuesTruncated(self):
        truncation = jnp.array([0., 0., 0., 1., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., 4_000., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(3_088.))

    def testComputeTDValuesReset(self):
        truncation = jnp.array([0., 0., 1., 0., 0.])
        termination = jnp.array([0., 0., 1., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., 4_000., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=0.9,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(3_250.))

    def testComputeUndiscountedTDValues(self):
        truncation = jnp.array([0., 0., 0., 0., 0.])
        termination = jnp.array([0., 0., 0., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(113_211))

    def testComputeUndiscountedTDValuesTerminated(self):
        truncation = jnp.array([0., 0., 0., 0., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(3_211.))

    def testComputeUndiscountedTDValuesTruncated(self):
        truncation = jnp.array([0., 0., 0., 1., 1.])
        termination = jnp.array([0., 0., 0., 1., 1.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., -999., 4_000., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(4_211))

    def testComputeUndiscountedTDValuesReset(self):
        truncation = jnp.array([0., 0., 1., 0., 0.])
        termination = jnp.array([0., 0., 1., 0., 0.])
        rewards = jnp.array([1., 10., 200., 3_000., 10_000.])
        values = jnp.array([-999., -999., 4_000., -999., -999.])
        bootstrap_value = jnp.array(100_000.)

        td_value = shac_losses.compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=1.,
        )

        np.testing.assert_approx_equal(td_value, jnp.array(4_011.))

if __name__ == '__main__':
    absltest.main()
