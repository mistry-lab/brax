from typing import Sequence, Tuple

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

import jax.numpy as jnp


@flax.struct.dataclass
class DiffRLSHACNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork

def make_inference_fn(shac_networks: DiffRLSHACNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      logits = shac_networks.policy_network.apply(*params, observations)
      return jnp.clip(logits, min=-0.1, max=0.1), {}

    return policy

  return make_policy


def make_shac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.elu,
    layer_norm: bool = True) -> DiffRLSHACNetworks:
  """Make SHAC networks with preprocessor."""
  policy_network = networks.make_policy_network(
      action_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm,
      dtype=jnp.float64)
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      layer_norm=layer_norm,
      dtype=jnp.float64)

  return DiffRLSHACNetworks(
      policy_network=policy_network,
      value_network=value_network)
