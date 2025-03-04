import jax
import jax.numpy as jnp

from mujoco import mjx
from brax.envs.fd.wrappers.terminal_reward import TerminalReward

def terminal_reward(dx: mjx.Data, action: jax.Array):
    pos_finger = dx.qpos[2]
    touch = dx.sensordata[0]
    p_finger = dx.sensordata[1:4]
    p_target = dx.sensordata[4:7]

    target_reward = -10. * jnp.sum((p_finger - p_target)**2)
    touch_reward = -4.* touch * pos_finger**2

    metrics = {
        'reward_target': target_reward,
        'reward_touch': touch_reward,
    }

    return TerminalReward(
        reward=target_reward,
        metrics=metrics
    )
