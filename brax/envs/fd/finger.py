from brax.envs.fd.fd_env import FDEnv

from etils import epath

import jax
import jax.numpy as jnp

from brax.io import mjcf
from brax.envs.base import State

class Finger(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/finger_mjx.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel", "ctrl", "sensordata"}, **kwargs)

    def _angle_axis_to_quaternion(self, angle_axis):
        """
        Converts an angle-axis vector to a quaternion.

        Args:
            angle_axis: A JAX array of shape (3,) representing the angle-axis vector.

        Returns:
            A JAX array of shape (4,) representing the quaternion [w, x, y, z].
        """
        theta = jnp.linalg.norm(angle_axis)

        def not_zero(theta, angle_axis):
            half_theta = 0.5 * theta
            k = jnp.sin(half_theta) / theta
            w = jnp.cos(half_theta)
            xyz = angle_axis * k
            quaternion = jnp.concatenate([jnp.array([w]), xyz])
            return quaternion

        def is_zero(theta, angle_axis):
            k = 0.5
            w = 1.0
            xyz = angle_axis * k
            quaternion = jnp.concatenate([jnp.array([w]), xyz])
            return quaternion

        quaternion = jax.lax.cond(
            theta > 1e-8,  # A small threshold to handle numerical precision
            not_zero,
            is_zero,
            theta,
            angle_axis
        )
        return quaternion

    def reset(self, rng: jax.Array, flag: bool = False) -> State:
        qpos_init = self._generate_initial_conditions(rng)
        qvel_init = jnp.zeros_like(qpos_init)  # or any distribution you like

        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_target': zero,
            'reward_ctrl': zero,
            'reward_touch': zero,
        }

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray, flag: bool = False) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        pos_finger = dx_next.qpos[2]
        u = dx_next.ctrl

        touch = dx_next.sensordata[0]
        p_finger = dx_next.sensordata[1:4]
        p_target = dx_next.sensordata[4:7]

        touch_reward = - 0.001 * touch * pos_finger **2

        if flag:
            state.metrics.update(
                reward_target = jnp.zeros_like(touch_reward),
                reward_ctrl = jnp.zeros_like(touch_reward),
                reward_touch = touch_reward
            )

            return state.replace(
                pipeline_state=dx_next, obs=obs, reward=reward
            )

        target_reward = - 0.1 * jnp.sum((p_finger - p_target)**2)
        ctrl_reward = - 0.00005 * jnp.sum(u ** 2)
        reward = target_reward + ctrl_reward + touch_reward

        state.metrics.update(
            reward_target = target_reward,
            reward_ctrl = ctrl_reward,
            reward_touch = touch_reward
        )

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )

    def _get_observation(self, dx: State):
        return jnp.concatenate([dx.qpos, dx.qvel, dx.sensordata])

    def _generate_initial_conditions(self, key: jax.Array) -> State:
        # Solution of IK
        _, key = jax.random.split(key)
        sign = 2.*jax.random.bernoulli(key, 0.5) - 1.

        # Reference target position in spinner local frame R_s
        _, key = jax.random.split(key, num=2)
        theta_l = jax.random.uniform(key, (1,), minval=0.6, maxval=2.5) # Polar Coord
        l_spinner = 0.22
        # r = jax.random.uniform(key, (1,), minval=l_spinner + 0.01, maxval=l_spinner + 0.01)
        r_l = l_spinner
        x_s,y_s = r_l*jnp.cos(theta_l), r_l*jnp.sin(theta_l) # Cartesian in R_l
        
        # Reference target position in finger frame R_f
        x, y = x_s, y_s - 0.39

        # Inverse kinematic formula
        l1, l2 = 0.17, 0.161
        q1 = sign * jnp.arccos( (x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2) )
        q0 = jnp.arctan2(y,x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2*jnp.cos(q1))

        # dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
        # dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))

        _, key = jax.random.split(key, num=2)
        theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)
        # dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))

        return jnp.concatenate([q0,q1,theta])
