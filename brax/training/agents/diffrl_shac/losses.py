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

import jax
import jax.numpy as jnp

from typing import Any

from brax import envs

from brax.training.agents.diffrl_shac import networks as shac_networks
from brax.training.batched_unroll import generate_batched_unroll
from brax.training.gae import compute_vs

from brax.training.types import Params, PRNGKey, ValueTrainingSample


def compute_td_value(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    discount: float = 0.99,
):
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(jax.lax.stop_gradient(bootstrap_value), 0)], axis=0
    )
    rewards_plus_values = rewards + \
        jax.lax.stop_gradient(discount * (1 - termination) * values_t_plus_1)
    rewards_plus_values *= jax.lax.stop_gradient(1 - truncation)

    acc = jnp.zeros_like(bootstrap_value)

    def sum_rewards(carry, target_t):
        reward_plus_value, truncation, termination, value = target_t
        carry = reward_plus_value + \
            jax.lax.stop_gradient(discount * (1 - termination)) * carry - \
            jax.lax.stop_gradient((1 - truncation) * value)
        return (carry), (carry)

    episode_reward, _ = jax.lax.scan(
        sum_rewards,
        acc,
        (
            rewards_plus_values,
            truncation,
            termination,
            values
        ),
        length=int(rewards.shape[0]),
        reverse=True,
    )

    return episode_reward + jax.lax.stop_gradient(values[0])

def make_losses(
    shac_network: shac_networks.DiffRLSHACNetworks,
    env: envs.Env,
    discounting: float,
    reward_scaling: float,
    gae_lambda: float,
    unroll_length: int,
    number: int
):
    value_apply = shac_network.value_network.apply
    make_policy = shac_networks.make_inference_fn(shac_network)

    """Creates the SHAC losses."""
    def critic_loss(
        value_params: Params,
        normalizer_params: Any,
        data: ValueTrainingSample
    ):
        # Put the time dimension first.
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        baseline = value_apply(normalizer_params, value_params, data.observation)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        bootstrap_value = value_apply(normalizer_params, value_params, terminal_obs)

        termination = (1 - data.discount) * (1 - data.truncation)

        vs = compute_vs(
            truncation=data.truncation,
            termination=termination,
            rewards=data.reward,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )

        # Value function loss
        v_error = vs - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        return v_loss, {}

    def actor_loss(
        policy_params: Params,
        value_params: Params,
        normalizer_params: Any,
        env_state: envs.State,
        key: PRNGKey,
    ):
        policy = make_policy((normalizer_params, policy_params))

        final_state, data = generate_batched_unroll(
            env=env,
            env_state=jax.lax.stop_gradient(env_state),
            policy=policy,
            key=key,
            unroll_length=unroll_length,
            number=number,
            reward_scaling=reward_scaling,
            extra_fields=('truncation', 'episode_metrics', 'episode_done'),
        )
        # Put the time dimension first.
        ordered_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        truncation = ordered_data.extras['state_extras']['truncation']

        values = value_apply(normalizer_params, value_params, ordered_data.observation)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], ordered_data.next_observation)
        bootstrap_value = value_apply(normalizer_params, value_params, terminal_obs)

        termination = (1 - ordered_data.discount) * (1 - truncation)

        loss = -compute_td_value(
            truncation=truncation,
            termination=termination,
            rewards=ordered_data.reward,
            values=values,
            bootstrap_value=bootstrap_value,
            discount=discounting
        )

        return jnp.mean(loss), {
            "final_state": final_state,
            "data": data
        }

    return critic_loss, actor_loss
