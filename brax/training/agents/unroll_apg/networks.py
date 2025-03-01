from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from flax.typing import Dtype

import jax.numpy as jnp

@flax.struct.dataclass
class UnrollAPGNetworks:
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution

def make_inference_fn(apg_networks: UnrollAPGNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey, step: int
    ) -> Tuple[types.Action, types.Extra]:
      logits = apg_networks.policy_network.apply(*params, observations, step)
      if deterministic:
        return apg_networks.parametric_action_distribution.mode(logits), {}
      return (
          apg_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

    return policy

  return make_policy


def make_apg_network(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.elu,
    layer_norm: bool = True,
    dtype: Dtype = 'float64') -> UnrollAPGNetworks:
  """Make SHAC networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm,
      dtype=jnp.dtype(dtype))

  return UnrollAPGNetworks(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution)
