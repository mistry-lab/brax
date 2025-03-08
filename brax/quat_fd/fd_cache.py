from typing import Callable, Optional

from dataclasses import dataclass
from mujoco import mjx
import jax.numpy as jnp

@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    num_u_dims: int
    eps: float
    sensitivity_mask: jnp.ndarray
    dx_flat_all_idx: Optional[jnp.ndarray]  # All indices (including free/ball quaternion)
    dx_flat_no_quat_idx: jnp.ndarray  # "normal" FD indices (excludes free/ball quaternion)
    dx_flat_quat_idx: Optional[jnp.ndarray] = None # subset that also lies in target_fields
    dx_flat_init_quat_idx: Optional[jnp.ndarray] = None # first index of each quaternion
    qpos_init_quat_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints
    quat_ijk_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints
