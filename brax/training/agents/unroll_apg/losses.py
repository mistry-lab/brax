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

from typing import Any

import jax
import jax.numpy as jnp

from brax import envs
from brax.training.agents.unroll_apg import networks as apg_networks
from brax.training import acting
from brax.training.types import Params

def compute_weighted_and_discounted_reward(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    discount: float = 0.99,
):
    truncated_rewards = rewards * jax.lax.stop_gradient(1 - truncation)

    def sum_rewards(carry, target_t):
        reward_plus_value, termination = target_t
        carry = reward_plus_value + \
            jax.lax.stop_gradient(discount * (1 - termination)) * carry
        return (carry), (carry)

    episode_reward, _ = jax.lax.scan(
        sum_rewards,
        jnp.zeros(rewards.shape[1:]),
        (
            truncated_rewards,
            termination,
        ),
        length=int(rewards.shape[0]),
        reverse=True,
    )

    return episode_reward

def compute_apg_loss(
    params: Params,
    normalizer_params: Any,
    env_state: envs.State,
    rng: jnp.ndarray,
    apg_network: apg_networks.UnrollAPGNetworks,
    env: envs.Env,
    include_time: bool,
    episode_length: int,
    number: int,
    entropy_cost: float,
    discounting: float,
    reward_scaling: float
):
    make_policy = apg_networks.make_inference_fn(apg_network)
    policy = make_policy((normalizer_params, params))

    def f(carry, unused_t):
        current_state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        next_state, data = acting.generate_unroll(
            env,
            current_state,
            policy,
            current_key,
            episode_length,
            extra_fields=('truncation', 'episode_metrics', 'episode_done', 'steps'),
            include_time=include_time,
        )
        return (next_state, next_key), data

    (next_state, _), data = jax.lax.scan(
        f,
        (env_state, rng),
        (),
        length=number,
    )

    # Have leading dimensions (number * num_envs, episode_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (episode_length,)

    # Put the time dimension first.
    ordered_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    truncation = ordered_data.extras['state_extras']['truncation']
    termination = (1 - ordered_data.discount) * (1 - truncation)

    # Entropy reward
    policy_logits = apg_network.policy_network.apply(
        normalizer_params,
        params,
        ordered_data.observation,
        jnp.expand_dims(ordered_data.extras['state_extras']['steps'], axis=-1),
    ) if include_time \
    else apg_network.policy_network.apply(
        normalizer_params,
        params,
        ordered_data.observation
    )

    entropy = jnp.mean(apg_network.parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    v_loss = -compute_weighted_and_discounted_reward(
        truncation=truncation,
        termination=termination,
        rewards=reward_scaling * ordered_data.reward,
        discount=discounting
    )

    total_loss = v_loss + jax.lax.stop_gradient(entropy_loss)

    return jnp.mean(total_loss), {
        "data": data,
        "next_state": next_state,
        "metrics": {
            'total_loss': total_loss,
            'v_loss': jnp.mean(v_loss),
            'entropy_loss': entropy_loss,
        },
    }
