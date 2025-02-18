
"""PPO tests."""

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs

import brax.training.agents.diffrl_shac as diffrl_shac

class SACTest(parameterized.TestCase):
  """Tests for SAC module."""

  def testTrain(self):
    """Test SAC with a simple env."""
    fast = envs.get_environment('fast')
    _, _, metrics = diffrl_shac.train(
        fast,
        num_timesteps=2**15,
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        discounting=0.99,
        batch_size=64,
        normalize_observations=True,
        reward_scaling=10,
        grad_updates_per_step=64,
        num_evals=3,
        seed=0,
    )
    self.assertGreater(metrics['eval/episode_reward'], 140 * 0.995)
    self.assertEqual(fast.reset_count, 3)  # type: ignore
    # once for prefill, once for train, once for eval
    self.assertEqual(fast.step_count, 3)  # type: ignore


if __name__ == '__main__':
  absltest.main()
