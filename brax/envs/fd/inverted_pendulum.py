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

from mujoco import mjx

from brax.io import mjcf
from brax.envs.fd.fd_env import FDEnv
from brax.envs.base import State
from etils import epath
import jax
import jax.numpy as jnp


class InvertedPendulum(FDEnv):
    def __init__(self,
            reward_shaping: bool = False,
            pole_angle_weight = 1.0,
            pole_velocity_weight = 0.1,
            cart_position_weight = 0.05,
            cart_velocity_weight = 0.1,
            ctrl_cost_weight=0.0,
            **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/inverted_pendulum.xml'
        sys = mjcf.load(path)

        self._reward_shaping = reward_shaping

        self._pole_angle_weight = pole_angle_weight
        self._pole_velocity_weight = pole_velocity_weight
        self._cart_position_weight = cart_position_weight
        self._cart_velocity_weight = cart_velocity_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl"}, **kwargs)
    
    def set_control(self, dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def reset(self, rng: jax.Array, flag: bool = False) -> State:
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
        reward, done, zeros = jnp.zeros(3)
        metrics = {
            "reward_pole_position": zeros,
            "reward_pole_velocity": zeros,
            "reward_cart_position": zeros,
            "reward_cart_velocity": zeros,
            "reward_ctrl": zeros,
        } if self._reward_shaping else {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array, flag: bool = False) -> State:
        """Run one timestep of the environment's dynamics."""

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        dx_next = self.step_fn(state.pipeline_state, action)
        obs = self._get_observation(dx_next)
        done = jnp.where(jnp.abs(obs[1]) > 0.2, 1.0, 0.0)

        if self._reward_shaping:
            pole_position_reward = \
                -self._pole_angle_weight * jnp.pow(dx_next.qpos[1], 2)
            pole_velocity_reward = \
                -self._pole_velocity_weight * jnp.pow(dx_next.qvel[1], 2)

            cart_position_reward = \
                -self._cart_position_weight * jnp.pow(dx_next.qpos[0], 2)
            cart_velocity_reward = \
                -self._cart_velocity_weight * jnp.pow(dx_next.qvel[0], 2)

            ctrl_reward = -self._ctrl_cost_weight * jnp.sum(jnp.square(action))

            state.metrics.update(
                reward_pole_position = cart_position_reward,
                reward_pole_velocity = cart_velocity_reward,
                reward_cart_position = pole_position_reward,
                reward_cart_velocity = pole_velocity_reward,
                reward_ctrl = ctrl_reward,
            )

            reward = cart_position_reward \
                + cart_velocity_reward \
                + pole_position_reward \
                + pole_velocity_reward \
                + ctrl_reward

            return state.replace(
                pipeline_state=dx_next, obs=obs, reward=reward, done=done
            )

        reward = 1.0
        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward, done=done
        )

    def _get_observation(self, pipeline_state: mjx.Data) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jnp.concatenate([pipeline_state.qpos, pipeline_state.qvel])
