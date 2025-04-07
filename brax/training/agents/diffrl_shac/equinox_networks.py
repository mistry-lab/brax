import equinox
import jax
import jax.numpy as jnp
import flax
from flax import linen
from brax.training import networks, types

from typing import Sequence, Tuple

class PolicyNet(equinox.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [equinox.nn.Linear(
            dims[i], dims[i + 1], key=keys[i], use_bias=True
        ) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x).squeeze()
        # x = jnp.tanh(x) * 1
        return x
    
@flax.struct.dataclass
class DiffRLSHACNetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork

def make_inference_fn(shac_networks: DiffRLSHACNetworks):
  """Creates params and inference function for the SHAC agent."""

  def make_policy(
      params: types.PolicyParams
  ) -> types.Policy:

    def policy(
        observations: types.Observation
    ) -> Tuple[types.Action, types.Extra]:
    #   model = equinox.combine(params)
      observations =  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), observations)
      logits = params[1](observations, 1)
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
    policy_network = PolicyNet([6, 128,256, 128, 2], key=jax.random.PRNGKey(0))

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
