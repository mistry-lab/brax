from brax.envs.base import Env

import abc

from mujoco import mjx
from typing import Optional, Set

import jax
import jax.numpy as jnp

from brax import base
from brax.fd.pipeline import build_fd_cache, make_step_fn
from brax.base import System

from typing import Mapping, Tuple, Union

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

def make_upscaled_data(mx: mjx.Data):
    dx_template = mjx.make_data(mx)
    dx_template = jax.tree.map(upscale, dx_template)
    return dx_template

class FDEnv(Env):
    @abc.abstractmethod
    def set_control(self, dx: mjx.Data, u: jnp.ndarray):
        pass

    def __init__(self, sys: System, target_fields: Optional[Set[str]] = None, eps: float = 1e-6, upscale=False):
        self.sys = sys
        self.mx = mjx.put_model(self.sys.mj_model)
        self.dx = mjx.make_data(self.mx)

        # dx_template = make_upscaled_data(self.mx)
        fd_cache = build_fd_cache(self.dx, target_fields, eps)
        self.step_fn = make_step_fn(self.mx, self.set_control, fd_cache)

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
        dx0 = mjx.make_data(self.mx)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

        return dx0
