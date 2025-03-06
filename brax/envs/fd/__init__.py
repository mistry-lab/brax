from brax.envs.fd import finger, linear, inverted_pendulum, pusher, shadow_hand, ant
from brax.envs.fd.fd_env import FDEnv

_envs = {
    'finger': finger.Finger,
    'linear': linear.Linear,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'pusher': pusher.Pusher,
    'shadow_hand': shadow_hand.ShadowHand,
    'ant': ant.Ant,
}

def get_environment(env_name: str, **kwargs) -> FDEnv:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)
