import jax

from mujoco import mjx
from brax.envs.fd.wrappers.terminal_reward import TerminalReward
from brax.envs.fd.two_body import TwoBody

def terminal_reward(dx: mjx.Data, action: jax.Array):

    return TerminalReward(
        reward=-10 * TwoBody.running_cost(dx, action),
        metrics={}
    )
