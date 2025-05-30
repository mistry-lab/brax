from typing import Sequence, Tuple

import jax

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from flax.typing import Dtype


@flax.struct.dataclass
class DiffRLSHACNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution

def make_inference_fn(
    shac_networks: DiffRLSHACNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(
      params: types.PolicyParams, deterministic: bool = False
  ) -> types.Policy:

    def policy(
        observations: types.Observation, key_sample: PRNGKey, step: int = None
    ) -> Tuple[types.Action, types.Extra]:
      logits = shac_networks.policy_network.apply(*params, observations, step) if step is not None \
        else shac_networks.policy_network.apply(*params, observations)
      if deterministic:
        return shac_networks.parametric_action_distribution.mode(logits), {}
      return (
          shac_networks.parametric_action_distribution.sample(
              logits, key_sample
          ),
          {},
      )

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
    layer_norm: bool = True,
    dtype: Dtype = 'float64') -> DiffRLSHACNetworks:
  """Make SHAC networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      kernel_init=jax.nn.initializers.variance_scaling(
        scale=0.1,
        mode="fan_in",
        distribution="uniform",
        dtype=dtype
      ),
      layer_norm=layer_norm,
      dtype=dtype)
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      # kernel_init=jax.nn.initializers.variance_scaling(
      #   scale=0.1,
      #   mode="fan_in",
      #   distribution="uniform",
      #   dtype=dtype
      # ),
      layer_norm=layer_norm,
      dtype=dtype)

  return DiffRLSHACNetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)
