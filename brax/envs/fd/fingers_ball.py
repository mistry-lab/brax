from brax.envs.fd.quat_fd_env import QuatFDEnv

from etils import epath

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat

from brax.io import mjcf
from brax.envs.base import State

class FingersBall(QuatFDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/fingers_ball.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel"}, **kwargs) 

    def reset(self, rng: jax.Array, flag: bool = False) -> State:
        qpos_init = self.dx.qpos
        qvel_init = self.dx.qvel

        u0 = jnp.asarray([0.1, 0.0, -0.1, 0.0])
        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {}

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray, flag: bool = False) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        reward = -FingersBall.running_cost(dx_next, u)

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )
    
    def _get_observation(self, dx: mjx.Data):
        return jnp.concatenate([dx.qpos, dx.qvel])

    @staticmethod
    def running_cost(dx: mjx.Data, action: jax.Array):
        quat_ref =   axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # Chordal distance :
        # it complies with the four metric requirements while being more numerically stable
        # and simpler than the geodesic distance
        # https://arxiv.org/pdf/2401.05396
        costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        return 0.01*costR + 0.0001*jnp.sum(dx.ctrl**2)
