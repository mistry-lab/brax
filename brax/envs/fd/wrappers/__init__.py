import jax

from brax.envs.fd.wrappers import terminal_reward_finger
from brax.envs.wrappers.terminal_reward import TerminalRewardEpisodeWrapper 

from brax.envs import Env

terminal_rewards = {
    'finger': terminal_reward_finger.terminal_reward,
}

def get_terminal_reward_wrapper(
        env: Env,
        terminal_reward_name: str,
        episode_length: int,
        action_repeat: int,
        vmap: bool) -> TerminalRewardEpisodeWrapper:

    terminal_reward = terminal_rewards[terminal_reward_name]
    if vmap:
        terminal_reward = jax.vmap(terminal_reward)

    return TerminalRewardEpisodeWrapper(
        env=env,
        terminal_reward=terminal_reward,
        episode_length=episode_length,
        action_repeat=action_repeat
    )
