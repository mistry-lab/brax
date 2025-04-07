import jax
import jax.numpy as jnp
from mujoco import mjx

from typing import Callable, Optional, Set
from jax.flatten_util import ravel_pytree
import numpy as np
from jax._src.util import unzip2

from brax.fd.fd_cache import FDCache
from brax.fd.base import State
from brax.base import Motion, System, Transform

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

def init(
        sys: System,
        q: jax.Array,
        qd: jax.Array,
        act: Optional[jax.Array] = None,
        ctrl: Optional[jax.Array] = None,
        unused_debug: bool = False,
    ) -> State:
    """Initializes physics data.

    Args:
        sys: a brax System
        q: (q_size,) joint angle vector
        qd: (qd_size,) joint velocity vector
        act: actuator activations
        ctrl: actuator controls
        unused_debug: ignored

    Returns:
        data: initial physics data
    """
    data = make_upscaled_data(sys)
    data = data.replace(qpos=q, qvel=qd)
    if act is not None:
        data = data.replace(act=act)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)

    data = mjx.forward(sys, data)

    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    return State(q=q, qd=qd, x=x, xd=xd, **data.__dict__)

def build_fd_cache(
    dx_ref: State,
    target_fields: Optional[Set[str]] = None,
    eps: float = 1e-6
) -> FDCache:
    """
    Build a cache containing:
      - Flatten/unflatten for dx_ref
      - The mask for relevant FD indices (e.g. qpos, qvel, ctrl)
      - The shape info for control
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel", "ctrl"}

    # Flatten dx
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]
    num_u_dims = dx_ref.ctrl.shape[0]

    # Gather leaves for qpos, qvel, ctrl
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
    sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
    indices = tuple(np.cumsum(sizes))

    idx_target_state = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        # Check if any level in the path has a 'name' that is in target_fields
        name_matches = any(
            getattr(level, 'name', None) in target_fields
            for level in path
        )
        if name_matches:
            idx_target_state.append(i)

    def leaf_index_range(leaf_idx):
        start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
        end = indices[leaf_idx]
        return np.arange(start, end)

    # Combine all relevant leaf sub-ranges
    inner_idx_list = []
    for i in idx_target_state:
        inner_idx_list.append(leaf_index_range(i))
    inner_idx = np.concatenate(inner_idx_list, axis=0)
    inner_idx = jnp.array(inner_idx, dtype=jnp.int32)

    # Build the sensitivity mask
    sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)

    return FDCache(
        unravel_dx=unravel_dx,
        sensitivity_mask=sensitivity_mask,
        inner_idx=inner_idx,
        dx_size=dx_size,
        num_u_dims=num_u_dims,
        eps=eps
    )

def make_step_fn(
    sys: System,
    set_control_fn: Callable,
    fd_cache: FDCache
) -> Callable:
    """
    Create a custom_vjp step function that takes (dx, u) and returns dx_next.
    We do finite differences (FD) in the backward pass using the info in fd_cache.
    """

    @jax.custom_vjp
    def step_fn(state: State, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(state, u)
        dx_next = mjx.step(sys, dx_with_ctrl)

        q, qd = dx_next.qpos, dx_next.qvel
        x = Transform(pos=dx_next.xpos[1:], rot=dx_next.xquat[1:])
        cvel = Motion(vel=dx_next.cvel[1:, 3:], ang=dx_next.cvel[1:, :3])
        offset = dx_next.xpos[1:, :] - dx_next.subtree_com[sys.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)

        return dx_next.replace(q=q, qd=qd, x=x, xd=xd)

    def step_fn_fwd(state: State, u: jnp.ndarray):
        dx_next = step_fn(state, u)
        return dx_next, (state, u, dx_next)

    def step_fn_bwd(res, g):
        """
        FD-based backward pass. We approximate d(dx_next)/d(dx,u) and chain-rule with g.
        Uses the cached flatten/unflatten info in fd_cache.
        """
        dx_in, u_in, dx_out = res

        # Convert float0 leaves in 'g' to zeros
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf

            return jax.tree_map(fix_leaf, diff_tree, grad_tree)

        mapped_g = map_g_to_dinput(dx_in, g)
        # jax.debug.print(f"mapped_g: {mapped_g}")
        g_array, _ = ravel_pytree(mapped_g)

        # Flatten dx_in, dx_out, and controls
        dx_array, _ = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)
        u_in_flat = u_in.ravel()

        # Grab cached info
        unravel_dx = fd_cache.unravel_dx
        sensitivity_mask = fd_cache.sensitivity_mask
        inner_idx = fd_cache.inner_idx
        num_u_dims = fd_cache.num_u_dims
        eps = fd_cache.eps

        # e = jnp.zeros_like(u_in_flat).at[i].set(eps)
        # u_in_eps_positive = (u_in_flat + e).reshape(u_in.shape)
        # dx_perturbed_positive = step_fn(dx_in, u_in_eps_positive)
        # dx_perturbed_array_positive, _ = ravel_pytree(dx_perturbed_positive)

        # u_in_eps_negative = (u_in_flat - e).reshape(u_in.shape)
        # dx_perturbed_negative = step_fn(dx_in, u_in_eps_negative)
        # dx_perturbed_array_negative, _ = ravel_pytree(dx_perturbed_negative)

        # =====================================================
        # =============== FD wrt control (u) ==================
       # =====================================================
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (num_u_dims, dx_dim)
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))

        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        # We only FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (len(inner_idx), dx_dim)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)

        # -----------------------------------------------------
        # Instead of scattering rows into a (dx_dim, dx_dim) matrix,
        # multiply Jx_rows directly with g_array[inner_idx].
        # This avoids building a large dense Jacobian in memory.
        # -----------------------------------------------------
        # Jx_rows[i, :] is derivative w.r.t. dx_array[inner_idx[i]].
        # We want sum_i [ Jx_rows[i] * g_array[inner_idx[i]] ].
        # => shape (dx_dim,)
        # Scatter those rows back to a full (dx_dim, dx_dim) matrix
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        dx_dim = dx_array.size

        # Solution 2 : Reduced size multiplication (inner_idx, inner_idx) @ (inner_idx,)
        d_x_flat_sub = Jx_rows[:, inner_idx] @ g_array[inner_idx]
        d_x_flat = scatter_rows(d_x_flat_sub, inner_idx, (dx_dim,))

        d_u = Ju_array[:, inner_idx] @ g_array[inner_idx]
        d_x = unravel_dx(d_x_flat)
        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn