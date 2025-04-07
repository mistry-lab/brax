from absl.testing import absltest

import jax
import jax.numpy as jnp
from brax.envs.base import State

from brax.envs.fd.wrappers.terminal_reward import TerminalReward
from brax.envs.wrappers.terminal_reward import TerminalRewardEpisodeWrapper
from brax.envs.fd import get_environment

class TrainingTest(absltest.TestCase):

    def test_episode_termination(self):
        env = get_environment('ant')
        terminal_reward = lambda x, y: TerminalReward(100., {})
        
        env = TerminalRewardEpisodeWrapper(
            env=env,
            terminal_reward=terminal_reward,
            episode_length=100,
            action_repeat=2,
        )

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        for i in range(100):
            state = env.step(state, jnp.zeros(8))


if __name__ == '__main__':
    absltest.main()
