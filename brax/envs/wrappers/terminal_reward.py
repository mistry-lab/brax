from typing import Callable

import jax
import jax.numpy as jnp

from brax.envs.base import Env, State
from brax.envs.wrappers.training import EpisodeWrapper

class TerminalRewardEpisodeWrapper(EpisodeWrapper):
    def __init__(self, env: Env, terminal_reward: Callable, episode_length: int, action_repeat: int):
        super().__init__(env, episode_length, action_repeat)

        self.terminal_reward = terminal_reward

        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def step(self, state: State, action: jax.Array, **kwargs) -> State:
        state = super().step(state, action, **kwargs)
        next_steps = state.info['steps'] + self.action_repeat

        state = state.replace(
            reward=jnp.where(
                next_steps >= self.episode_length,
                self.terminal_reward(state.pipeline_state, action).reward,
                state.reward,
            ),
            metrics=state.metrics
        )

        return state

    def reset(self, rng: jax.Array) -> State:
        return super().reset(rng)
