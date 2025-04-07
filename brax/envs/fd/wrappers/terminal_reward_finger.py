import jax
import jax.numpy as jnp

from mujoco import mjx
from brax.envs.fd.wrappers.terminal_reward import TerminalReward

def _terminal_reward_value(qpos, sensordata, action: jax.Array):
    pos_finger = qpos[2]
    touch = sensordata[0]
    p_finger = sensordata[1:4]
    p_target = sensordata[4:7]

    #dist_reward = - 10 * jnp.sum((p_finger - p_target)**2)
    target_reward = - (pos_finger + jnp.pi / 2) ** 2

    return target_reward

@jax.custom_vjp
def terminal_reward_value(qpos, sensordata, action: jax.Array):
    return _terminal_reward_value(qpos, sensordata, action)

def _surrogate_reward(qpos, sensordata, action: jax.Array):
    pos_finger = qpos[2]
    touch = sensordata[0]
    p_finger = sensordata[1:4]
    p_target = sensordata[4:7]

    target_reward = - 1_000 * jnp.sum((p_finger - p_target)**2)

    return 0.0 # target_reward

surrogate_derivative = jax.grad(_surrogate_reward, argnums=(0, 1, 2))

def terminal_reward(dx: mjx.Data, action: jax.Array):
    reward = terminal_reward_value(dx.qpos, dx.sensordata, action)

    return TerminalReward(
        reward=reward,
        metrics={}
    )

def _get_reward_forward(qpos, sensordata, action: jax.Array):
    reward = _terminal_reward_value(qpos, sensordata, action)
    return reward, (qpos, sensordata, action, reward)

def _get_reward_backward(res, g):
    qpos_in, sensordata_in, action_in, reward = res
    
    d_qpos, d_sensordata, d_u = surrogate_derivative(
        qpos_in,
        sensordata_in,
        action_in
    )

    return (
        d_qpos * g,
        d_sensordata * g,
        d_u * g,
    )

terminal_reward_value.defvjp(_get_reward_forward, _get_reward_backward)
