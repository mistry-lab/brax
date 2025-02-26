# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpointing for SHAC."""

import json
from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from brax.training.agents.unroll_apg import networks as apg_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'unroll_apg_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[Union[apg_networks.UnrollAPGNetworks]],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_unroll_apg_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[apg_networks.UnrollAPGNetworks],
) -> apg_networks.UnrollAPGNetworks:
  """Generates a PPO network given config."""
  return checkpoint.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        apg_networks.UnrollAPGNetworks
    ] = apg_networks.make_apg_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from SHAC checkpoint."""
  path = epath.Path(path)

  config_path = path.parent / _CONFIG_FNAME
  if not config_path.exists():
    raise ValueError(f'SHAC config file not found at {config_path.as_posix()}')

  config = config_dict.create(**json.loads(config_path.read_text()))

  params = load(path)
  shac_network = _get_unroll_apg_network(config, network_factory)
  make_inference_fn = apg_networks.make_inference_fn(shac_network)

  return make_inference_fn(params, deterministic=deterministic)
