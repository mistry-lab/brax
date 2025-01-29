# Copyright 2022 The Brax Authors.
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

"""Short-Horizon Actor Critic.

See: https://arxiv.org/pdf/2204.07137.pdf
"""

import functools

import jax
import jax.numpy as jnp
import torch

from brax import envs
from typing import Any

from brax.training.agents.shac import networks as shac_networks

from brax.training import types
from brax.training.types import Params
import flax


@flax.struct.dataclass
class SHACNetworkParams:
  """Contains training state for the learner."""
  policy: Params
  value: Params

def compute_vs_minus_v_xs(discount_factor, acc_delta, target_t):
    delta, termination = target_t
    acc_delta = delta + discount_factor * (1 - termination) * acc_delta
    return (acc_delta), (acc_delta)

def compute_actor_loss(
    num_envs: int,
    ret_rms: bool,
    discount_factor: float,
    shac_network: shac_networks.SHACNetworks,
    value_params: Params,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
):
    value_apply = shac_network.value_network.apply

    values = value_apply(normalizer_params, value_params, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_values = value_apply(normalizer_params, value_params, terminal_obs)

    truncation = data.extras['state_extras']['truncation']

    termination = (1 - data.discount) * (1 - truncation)
    truncation_mask = 1 - truncation
    # Append terminal values to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(terminal_values, 0)], axis=0)

    deltas = data.reward + discount_factor * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros(num_envs)
    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        functools.partial(compute_vs_minus_v_xs, discount_factor),
        (acc),
        (deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    # Entropy reward
    total_loss = policy_loss

    return total_loss, {
        'policy_loss': policy_loss
    }

def compute_critic_loss(
    params: Params,
    normalizer_params: Any,
    data: types.Transition,
    shac_network: shac_networks.SHACNetworks,
):
    value_apply = shac_network.value_network.apply
    values = value_apply(normalizer_params, params, data.observation)

    target_values = batch_sample["target_values"]
    critic_loss = ((predicted_values - target_values) ** 2).mean()

    return critic_loss

@torch.no_grad()
def compute_target_values(num_envs: int,
    critic_method: str,
    shac_network: shac_networks.SHACNetworks,
    params: Params,
    normalizer_params: Any,
    data: types.Transition,
    gamma: float,
    steps_num: int):

    value_apply = shac_network.value_network.apply

    values = value_apply(normalizer_params, params, data.observation)
    terminal_value = value_apply(normalizer_params, params, data.next_observation[-1])
    rewards = data.reward * reward_scaling

    values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(terminal_value, 0)], axis=0)

    if critic_method == "one-step":
        return rewards + gamma * values_t_plus_1
    elif critic_method == "td-lambda":
        Ai = torch.zeros(num_envs, dtype=torch.float32)
        Bi = torch.zeros(num_envs, dtype=torch.float32)
        lam = torch.ones(num_envs, dtype=torch.float32)
        for i in reversed(range(steps_num)):
            lam = lam * lam * (1.0 - done_mask[i]) + done_mask[i]
            Ai = (1.0 - done_mask[i]) * (
                lam * gamma * Ai
                + gamma * next_values[i]
                + (1.0 - lam) / (1.0 - lam) * rew_buf[i]
            )
            Bi = (
                gamma
                * (
                    next_values[i] * done_mask[i]
                    + Bi * (1.0 - done_mask[i])
                )
                + rew_buf[i]
            )
            target_values[i] = (1.0 - lam) * Ai + lam * Bi
    else:
        raise NotImplementedError
