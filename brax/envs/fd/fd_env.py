from brax.envs.base import Env

import abc

from mujoco import mjx
from typing import Optional, Set

import jax.numpy as jnp

from brax.fd.pipeline import build_fd_cache, make_step_fn
from brax.fd.upscale import make_upscaled_data

class FDEnv(Env):
    @abc.abstractmethod
    def set_control(self, dx: mjx.Data, u: jnp.ndarray):
        pass

    def __init__(self, model: mjx.Model, target_fields: Optional[Set[str]] = None, eps: float = 1e-6):
        self.mx = mjx.put_model(model)
        dx_template = make_upscaled_data(self.mx)
        fd_cache = build_fd_cache(dx_template, target_fields, eps)
        self.step_fn = make_step_fn(self.mx, self.set_control, fd_cache)

    @property
    def backend(self):
        return "fd"
