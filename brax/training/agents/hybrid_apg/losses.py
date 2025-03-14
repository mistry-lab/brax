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

from typing import Any, Tuple

from brax import envs
from brax.training.agents.hybrid_apg import networks as apg_networks
from brax.training.batched_unroll import generate_batched_unroll
from brax.training.types import Metrics, Params, PRNGKey, Transition

from brax.training.gae import compute_gae

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
    apg_network: apg_networks.HybridAPGNetworks,
    env: envs.Env,
    entropy_cost: float,
    discounting: float,
    reward_scaling: float,
    gae_lambda: float,
    clipping_epsilon: float,
    normalize_advantage: bool,
    unroll_length: int,
    number: int
):
    value_apply = apg_network.value_network.apply
    make_policy = apg_networks.make_inference_fn(apg_network)

    """Creates the SHAC losses."""
    def compute_ppo_loss(
        policy_params: Params,
        value_params: Params,
        normalizer_params: Any,
        data: Transition,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Metrics]:
        """Computes PPO loss.

        Args:
            params: Network parameters,
            normalizer_params: Parameters of the normalizer.
            data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
            ['policy_extras']['log_prob']
            rng: Random key
            ppo_network: PPO networks.
            entropy_cost: entropy cost.
            discounting: discounting,
            reward_scaling: reward multiplier.
            gae_lambda: General advantage estimation lambda.
            clipping_epsilon: Policy loss clipping epsilon
            normalize_advantage: whether to normalize advantage estimate

        Returns:
            A tuple (loss, metrics)
        """
        parametric_action_distribution = apg_network.parametric_action_distribution
        policy_apply = apg_network.policy_network.apply
        value_apply = apg_network.value_network.apply

        # Put the time dimension first.
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        policy_logits = policy_apply(
            normalizer_params,
            policy_params,
            data.observation,
            jnp.expand_dims(data.extras['state_extras']['steps'], axis=-1)
        )

        baseline = value_apply(normalizer_params, value_params, data.observation)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        bootstrap_value = value_apply(normalizer_params, value_params, terminal_obs)

        rewards = data.reward * reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        target_action_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras['policy_extras']['raw_action']
        )
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

        vs, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (
            jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
        )

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
        entropy_loss = entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss
        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss,
        }

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
            extra_fields=('truncation', 'episode_metrics', 'episode_done', 'steps'),
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

    return compute_ppo_loss, actor_loss
