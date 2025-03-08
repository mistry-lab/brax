from brax.envs.fd.fd_env import FDEnv

from etils import epath

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat

from brax.io import mjcf
from brax.envs.base import State

class FingersBall(FDEnv):
    def __init__(self, **kwargs):
        path = epath.resource_path('brax') / 'envs/assets/fd/fingers_ball.xml'
        sys = mjcf.load(path)
        super().__init__(sys=sys, target_fields={"qpos", "qvel"}, **kwargs) 

    def reset(self, rng: jax.Array, upscale=False) -> State:
        qpos_init = self._generate_initial_conditions(rng)
        qvel_init = jnp.zeros_like(qpos_init)  # or any distribution you like

        dx0 = self.pipeline_init(qpos_init, qvel_init)
        obs = self._get_observation(dx0)

        reward, done, zero = jnp.zeros(3)
        metrics = {}

        return State(dx0, obs, reward, done, metrics)

    def step(self, state: State, u: jnp.ndarray) -> State:
        dx_next = self.step_fn(state.pipeline_state, u)
        obs = self._get_observation(dx_next)

        reward = -FingersBall.running_cost(dx_next, u)

        return state.replace(
            pipeline_state=dx_next, obs=obs, reward=reward
        )
    
    def _get_observation(self, dx: mjx.Data):
        return jnp.concatenate([dx.qpos, dx.qvel])

    def _generate_initial_conditions(self, rng: jax.Array) -> State:
        idata = mujoco.MjData(self.sys.mj_model)
        q0, q1, q2, q3 = -0.4, 0.44, 0.44, -0.4
        idata.qpos[0], idata.qpos[1], idata.qpos[2], idata.qpos[3] = q0, q1, q2, q3
        Nlength = 200

        u0 = 0.5 * jax.random.normal(jax.random.PRNGKey(0), (Nlength, 4))
        u0 = u0.at[:, 0].set(0.1)
        u0 = u0.at[:, 2].set(-0.1)
        u0 = u0.at[:, 3].set(0.)
        u0 = u0.at[:, 1].set(0.)
        qpos = jnp.array(idata.qpos)

        return u0

    @staticmethod
    def running_cost(dx: mjx.Data, action: jax.Array):
        quat_ref =   axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # Chordal distance :
        # it complies with the four metric requirements while being more numerically stable
        # and simpler than the geodesic distance
        # https://arxiv.org/pdf/2401.05396
        costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        return 0.01*costR + 0.0001*jnp.sum(dx.ctrl**2)

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx
