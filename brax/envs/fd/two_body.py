import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat

from brax.envs.fd.quat_fd_env import QuatFDEnv

from etils import epath

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat

from brax.io import mjcf
from brax.envs.base import State

class TwoBody(QuatFDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/two_body.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, ctrl_dim=1, target_fields={"qpos", "qvel"}, **kwargs) 

    def reset(self, rng: jax.Array, flag: bool = False) -> State:
        qpos_init = self.dx.qpos
        qvel_init = self.dx.qvel

        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {}

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray, flag: bool = False) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        reward = -TwoBody.running_cost(dx_next, u)

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )

    def _get_observation(self, dx: mjx.Data):
        return jnp.concatenate([dx.qpos, dx.qvel])

    @staticmethod
    def running_cost(dx: mjx.Data, action: jax.Array):
        quat_ref =   axis_angle_to_quat(jnp.array([0.,1.,0.]), jnp.array([2.35]))
        # Chordal distance :
        # it complies with the four metric requirements while being more numerically stable
        # and simpler than the geodesic distance
        # https://arxiv.org/pdf/2401.05396
        costR = jnp.sum((quat_to_mat(dx.qpos[0:4])  - quat_to_mat(quat_ref))**2)

        # Geodesic distance : Log of the diff in rotation matrix, then skew-symmetric extraction, then norm.
        # It defines a smooth metric since both the logarithmic map and the Euclidean norm are smooth.
        # However, it brings more computational expense and numerical instability
        # from the logarithm map for small rotations
        # Note : Arccos function can be used but will only compute the abs value of the norm, might be problematic to get the
        # sign or direction for the gradient.
        # costR = rotation_distance(quat_to_mat(dx.qpos[3:7]),quat_to_mat(quat_ref))
        # R0 = quat_to_mat(quaternion_multiply(quaternion_conjugate(dx.qpos[3:7]), quat_ref))
        # error = jnp.tanh((jnp.trace(R0) -1)/2)
        # error = jnp.sum(R0,axis = -1)
        # error = jnp.arccos((jnp.trace(R0) -1)/2)
        # error = (R0 - jnp.identity(3))**2
        # error = jnp.sum((R0)**2)

        # return 0.01*jnp.array([jnp.sum(quat_diff**2)]) + 0.00001*dx.qfrc_applied[5]**2
        return 0.001*costR + 0.000001*dx.qfrc_applied[2]**2 + 0.000001*dx.qvel[2]**2
        # return 0.00001 * dx.qfrc_applied[5] ** 2

    def set_control(self, dx, u):
        dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[1].set(u[0]))
        return dx

    @property
    def action_size(self) -> int:
        return 1
