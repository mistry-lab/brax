from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

import jax.numpy as jnp


@flax.struct.dataclass
class UnrollAPGNetworks:
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution

def make_inference_fn(shac_networks: UnrollAPGNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = shac_networks.policy_network.apply(*params, observations)
      return jnp.clip(logits, -0.01, 0.01), {}

    return policy

  return make_policy


def make_apg_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.elu,
    layer_norm: bool = True) -> UnrollAPGNetworks:
  """Make SHAC networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm)

  return UnrollAPGNetworks(
      policy_network=policy_network,
      parametric_action_distribution=parametric_action_distribution)
