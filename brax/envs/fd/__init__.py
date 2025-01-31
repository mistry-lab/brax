from brax.envs.fd import finger
from brax.envs.fd.fd_env import FDEnv

_envs = {
    'finger': finger.Finger,
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
