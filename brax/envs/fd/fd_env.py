from brax.envs.base import Env

from mujoco import mjx
from typing import Optional, Set

import jax

from brax.fd.pipeline import build_fd_cache, make_step_fn, init, make_upscaled_data
from brax.fd.base import State
from brax.base import System
from brax import base

from typing import Mapping, Tuple, Union

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

class FDEnv(Env):
    def __init__(self, sys: System, target_fields: Optional[Set[str]] = None, eps: float = 1e-6, upscale=False):
        self.sys = sys
        self.mx = mjx.put_model(self.sys.mj_model)
        dx = make_upscaled_data(self.mx)

        self.dx = init(self.sys, dx.qpos, dx.qvel)

        fd_cache = build_fd_cache(self.dx, target_fields, eps)
        self.step_fn = make_step_fn(self.sys, self.set_control, fd_cache)

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

    def pipeline_init(self, qpos_init: jax.Array, qvel_init: jax.Array) -> mjx.Data:
        return init(self.sys, qpos_init, qvel_init)
