from brax.envs.fd.fd_env import FDEnv

import mujoco
from mujoco import mjx

from etils import epath

import jax
import jax.numpy as jnp
from brax.fd.upscale import make_upscaled_data
from brax.envs.base import State

from typing import Mapping, Tuple, Union

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

class Linear(FDEnv):
    def __init__(self, eps: float = 1e-6):
        pass

    @property
    def observation_size(self) -> ObservationSize:
        return 1

    @property
    def action_size(self) -> int:
        return 1

    def reset(self, rng: jax.Array) -> State:
        x0 = 1.0

        reward, done, truncation = jnp.zeros(3)
        metrics = {}
        info = {"truncation": truncation}

        return State(x0, jnp.array([x0]), reward, done, metrics, info)

    def step(self, state: State, u: jnp.ndarray) -> State:
        x_next = 0.5 * state.pipeline_state + u
        reward = - self.running_cost(x_next, u)

        return state.replace(
            pipeline_state=x_next, obs=jnp.array([x_next]), reward=reward
        )

    def running_cost(self, x, u):
        return x + u 

    def terminal_cost(x):
        return x

    def set_control(self, dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
