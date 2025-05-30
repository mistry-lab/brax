from brax.envs.base import Env

from mujoco import mjx
from typing import Optional, Set

import jax

import brax.quat_fd.pipeline as fd_pipeline

from brax.fd.base import State
from brax.base import System

from typing import Mapping, Tuple, Union

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

class QuatFDEnv(Env):
    def __init__(self, sys: System, ctrl_dim: int = None, target_fields: Optional[Set[str]] = None, eps: float = 1e-6):
        self.sys = sys
        self.mx = mjx.put_model(self.sys.mj_model)
        dx = fd_pipeline.make_upscaled_data(self.mx)

        self.dx = fd_pipeline.init(self.sys, dx.qpos, dx.qvel)

        if ctrl_dim is None:
            ctrl_dim = self.action_size

        fd_cache = fd_pipeline.build_fd_cache(self.sys, self.dx, target_fields, ctrl_dim, eps)
        self.step_fn = fd_pipeline.make_step_fn(self.sys, self.set_control, fd_cache)

    def set_control(self, dx: State, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    @property
    def backend(self):
        return "fd"

    @property
    def action_size(self) -> int:
        return self.sys.act_size()

    @property
    def observation_size(self) -> ObservationSize:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        obs = reset_state.obs
        if isinstance(obs, jax.Array):
            return obs.shape[-1]
        return jax.tree_util.tree_map(lambda x: x.shape, obs)

    def pipeline_init(self, qpos_init: jax.Array, qvel_init: jax.Array, ctrl: jax.Array = None) -> mjx.Data:
        return fd_pipeline.init(self.sys, qpos_init, qvel_init, ctrl)
