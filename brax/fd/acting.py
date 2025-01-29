import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox

from brax.fd.upscale import upscale

@equinox.filter_jit
def simulate_trajectories(
        mx,
        qpos_inits,  # shape (B, nq) for example
        qvel_inits,  # shape (B, nqvel)
        running_cost_fn,
        terminal_cost_fn,
        step_fn,
        params,
        static,
        length,
        keys
):
    """
    Simulate a *batch* of trajectories (B of them) with a single policy.

    Args:
      mx: MuJoCo model container.
      qpos_inits: shape (B, n_qpos)
      qvel_inits: shape (B, n_qvel)
      running_cost_fn, terminal_cost_fn: same cost structure as before.
      step_fn: the custom FD-based step function returned by make_step_fn.
      params, static: your policy parameters & static parts (from equinox.partition).
      length: number of steps per trajectory.
    Returns:
      - states_batched: shape (B, length, 2 * n_qpos) (if that’s your total state dimension)
      - total_cost: a scalar cost (mean or sum across the batch).
    """
    # Combine the param & static into the actual model (same as simulate_trajectory).
    model = equinox.combine(params, static)

    def single_trajectory(qpos_init, qvel_init, key):
        """Simulate one trajectory given a single (qpos_init, qvel_init)."""
        # Build the initial MuJoCo data
        dx0 = mjx.make_data(mx)
        dx0 = jax.tree_map(upscale, dx0)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

        # Define the scanning function for a single rollout
        def scan_step_fn(carry, _):
            dx, key = carry
            key, subkey = jax.random.split(key)

            x = jnp.concatenate([dx.qpos, dx.qvel])
            
            # Add noise to the control  
            noise = 0.2 * jax.random.normal(subkey, mx.nu)
            # jax.debug.print("noise : {}", noise)
            u = model(x, dx.time) + noise # policy output

            dx = step_fn(dx, u)  # FD-based MuJoCo step
            c = running_cost_fn(dx)
            state = jnp.concatenate([dx.qpos, dx.qvel])
            return (dx,key), (state, c)

        key, subkey = jax.random.split(key)
        (dx_final, _), (states, costs) = jax.lax.scan(scan_step_fn, (dx0,subkey), length=length)
        total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
        return states, total_cost

    # vmap across the batch dimension
    states_batched, costs_batched = jax.vmap(single_trajectory)(qpos_inits, qvel_inits, keys)

    total_cost = jnp.mean(costs_batched)

    return states_batched, total_cost
