from absl.testing import parameterized
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from brax.training.agents.diffrl_shac import losses as shac_losses

class DiffRLLossesTest(parameterized.TestCase):
    def testComputeVsMinusVxs(self):
        acc_delta = 1.0
        # (deltas, termination)
        target_t = (jnp.array([11.0, -3.0]), jnp.array([0.0, 1.0]))

        gam, rew_acc = shac_losses.compute_vs_minus_v_xs(0.9, acc_delta, target_t)
        np.testing.assert_array_equal(rew_acc, jnp.array([11.0, 0.0]))
        np.testing.assert_array_equal(gam, jnp.array([0.9, 1.0]))

    def testComputeActorLoss(self):
        pass

if __name__ == '__main__':
    absltest.main()
