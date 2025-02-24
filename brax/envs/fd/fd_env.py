from brax.envs.base import Env

import abc

from mujoco import mjx
from typing import Optional, Set

import jax
import jax.numpy as jnp

from brax import base
from brax.fd.pipeline import build_fd_cache, make_step_fn
from brax.fd.upscale import make_upscaled_data
from brax.envs.base import State

import brax.mjx.pipeline as mjx_pipeline

from typing import Mapping, Tuple, Union

ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]

class FDEnv(Env):
    @abc.abstractmethod
    def set_control(self, dx: mjx.Data, u: jnp.ndarray):
        pass

    def __init__(self, sys: State, target_fields: Optional[Set[str]] = None, eps: float = 1e-6, upscale=False):
        self.sys = sys        

        dx_template = make_upscaled_data(self.sys) if upscale else mjx.make_data(self.sys)
        fd_cache = build_fd_cache(dx_template, target_fields, eps)
        self.step_fn = make_step_fn(self.sys, self.set_control, fd_cache)

    @property
    def backend(self):
        return "fd"

    @property
    def observation_size(self) -> ObservationSize:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        obs = reset_state.obs
        if isinstance(obs, jax.Array):
            return obs.shape[-1]
        return jax.tree_util.tree_map(lambda x: x.shape, obs)

    def pipeline_init(
        self,
        q: jax.Array,
        qd: jax.Array,
        act: Optional[jax.Array] = None,
        ctrl: Optional[jax.Array] = None,
    ) -> base.State:
        """Initializes the pipeline state."""
        return mjx_pipeline.init(self.sys, q, qd, act, ctrl)
