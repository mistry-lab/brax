from typing import Optional, Callable, Tuple

import functools

from brax import envs
from brax import base
from brax.training.agents.unroll_apg import networks as apg_networks
from brax.training.agents.unroll_apg import losses as apg_losses

from brax.training import types, acting
from brax.training.types import Params
from brax.training import gradients
from brax.training.acme import running_statistics, specs

import jax
import flax
import jax.numpy as jnp

import logging

import optax

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

def step(
    environment: envs.Env,
    network_factory: types.NetworkFactory[
              apg_networks.UnrollAPGNetworks] = apg_networks.make_apg_network,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.,
    grad_norm: Optional[float] = None,
    actor_lr: float = 2e-3,
    batch_size: int = 32,
    betas: Tuple[float, float] = (0.7, 0.95),
    num_envs: int = 1,
    normalize_observations: bool = False,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    rng = jax.random.PRNGKey(1)
    rng, key_eval, key_policy, env_key = jax.random.split(rng, 4)

    env = envs.training.wrap(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    apg_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize)

    loss_fn = functools.partial(apg_losses.compute_apg_loss,
        apg_network=apg_network,
        env=env,
        episode_length=episode_length,
        number=batch_size // num_envs,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
    )

    optimizer = optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
    if grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm),
            optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
        )

    update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, pmap_axis_name=None, has_aux=True)

    obs_size = env.observation_size
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float64'))
    )

    params = apg_network.policy_network.init(key_policy)
    optimizer_state = optimizer.init(params)

    env_keys = jax.random.split(env_key, num_envs)
    env_state = env.reset(env_keys)

    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=jnp.zeros(()),
    )

    make_policy = apg_networks.make_inference_fn(apg_network)
    evaluator = acting.Evaluator(
        env,
        functools.partial(make_policy),
        num_eval_envs=num_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=key_eval)

    metrics = evaluator.run_evaluation(
        (
            training_state.normalizer_params,
            training_state.params
        ),
        training_metrics={},
    )
    logging.info(metrics)

    for i in range(100):
        print(params['params']['hidden_0']['kernel'].dtype)
        print(normalizer_params.mean.dtype)

        (loss, extras), params, optimizer_state = update_fn(
            params,
            normalizer_params,
            env_state,
            rng,
            optimizer_state=optimizer_state
        )

        env_state = extras["next_state"]
        data = extras["data"]
        metrics = extras["metrics"]

        # normalizer_params = running_statistics.update(
        #     training_state.normalizer_params,
        #     data.observation,
        # )

        training_state = training_state.replace(
            params=params,
            optimizer_state=optimizer_state,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps,
        )

        metrics = evaluator.run_evaluation(
            (
                training_state.normalizer_params,
                training_state.params
            ),
            training_metrics=metrics,
        )
        logging.info(metrics)

        print(loss)
