# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""An inverted pendulum environment."""

from brax import base
from brax.io import mjcf
from brax.envs.fd.fd_env import FDEnv
from brax.envs.base import State
from etils import epath
import jax
import jax.numpy as jnp


class InvertedPendulum(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/inverted_pendulum.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl"}, **kwargs)

    @property
    def action_size(self):
        return 1
    
    def set_control(self, dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_observation(pipeline_state)
        reward, done = jnp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        dx_next = self.step_fn(state.pipeline_state, action)
        obs = self._get_observation(dx_next)
        reward = -self._running_cost(dx_next)
        done = jnp.where(jnp.abs(obs[1]) > 0.2, 1.0, 0.0)
        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward, done=done
        )

    def _get_observation(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    def _running_cost(self, dx):
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2)
