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

class Finger(FDEnv):
    def __init__(self, eps: float = 1e-6):
        path = epath.resource_path('brax') / 'envs/assets/fd/finger_mjx.xml'
        self.model = mujoco.MjModel.from_xml_path(str(path))
        super().__init__(self.model, {"qpos", "qvel", "ctrl"}, eps)

    @property
    def observation_size(self) -> ObservationSize:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        obs = reset_state.obs
        if isinstance(obs, jax.Array):
            return obs.shape[-1]
        return jax.tree_util.tree_map(lambda x: x.shape, obs)

    @property
    def action_size(self) -> int:
        return self.model.nu

    def _angle_axis_to_quaternion(self, angle_axis):
        """
        Converts an angle-axis vector to a quaternion.

        Args:
            angle_axis: A JAX array of shape (3,) representing the angle-axis vector.

        Returns:
            A JAX array of shape (4,) representing the quaternion [w, x, y, z].
        """
        a0, a1, a2 = angle_axis
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

    def reset(self, rng: jax.Array, upscale=False) -> State:
        qpos_init = self.generate_initial_conditions(rng)
        qvel_init = jnp.zeros_like(qpos_init)  # or any distribution you like

        dx0 = make_upscaled_data(self.mx) if upscale else mjx.make_data(self.mx)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {}

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        #reward = jnp.where(state.info["truncation"], -self.terminal_cost(dx_next), -self.running_cost(dx_next))
        reward = -self.running_cost(dx_next)

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )

    def _get_observation(self, dx: mjx.Data):
        return jnp.concatenate([dx.qpos, dx.qvel])

    def generate_initial_conditions(self, rng: jax.Array) -> State:
        # Solution of IK
        _, key = jax.random.split(rng)
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
        q1 = sign * (x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2)
        q0 = jnp.arctan2(y,x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2*jnp.cos(q1))

        # dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
        # dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))

        _, key = jax.random.split(key, num=2)
        theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)
        # dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))

        return jnp.concatenate([q0,q1,theta])

    def running_cost(self, dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl

        return  0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

    def terminal_cost(self, dx):
        pos_finger = dx.qpos[2]

        touch = dx.sensordata[0]
        p_finger = dx.sensordata[1:4]
        p_target = dx.sensordata[4:7]
        return  10. * jnp.sum((p_finger - p_target)**2) + 4.* touch * pos_finger**2

    def set_control(self, dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))
