import jax
import jax.numpy as jnp

from mujoco import mjx

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
