from typing import Optional, Callable, Tuple, Union

import functools

from brax import envs
from brax import base
from brax.training.agents.unroll_apg import networks as apg_networks
from brax.training.agents.unroll_apg import losses as apg_losses

from brax.training import types, acting
from brax.training.types import Params, PRNGKey
from brax.training import gradients
from brax.training.acme import running_statistics, specs
from brax.v1 import envs as envs_v1

import jax
import flax
import jax.numpy as jnp

import logging

import optax

_PMAP_AXIS_NAME = 'i'

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
      loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    obs_size = env.observation_size
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float64'))
    )

    params = apg_network.policy_network.init(key_policy)
    optimizer_state = optimizer.init(params)

    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count

    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(
        env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
    )
    env_state = jax.vmap(env.reset)(env_keys)

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
        key=key_eval,
        include_time=True)

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params,
        )),
        training_metrics={},
    )
    logging.info(metrics)

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState,
        Union[envs.State, envs_v1.State],
        types.Metrics,
    ]:
        # train actor
        (loss, extras), params, optimizer_state = update_fn(
            training_state.params,
            training_state.normalizer_params,
            env_state,
            key,
            optimizer_state=training_state.optimizer_state
        )

        data = extras["data"]
        next_state = extras["next_state"]
        metrics = extras["metrics"]

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        training_state = training_state.replace(
            params=params,
            optimizer_state=optimizer_state,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps,
        )

        return training_state, next_state, metrics

    def training_epoch(unroll_length, training_state, env_state, k):
        
        k, new_key = jax.random.split(k)
        training_state, next_state, metrics = training_step(training_state, env_state, new_key)

        return training_state, next_state, metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        in_axes=(None, 0, 0, 0)
    )

    def training_epoch_with_timing(training_state, env_state, rng):
        training_state, env_state, metrics = training_epoch(10, training_state, env_state, rng)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params,
            )),
            training_metrics=metrics,
        )
        logging.info(metrics)

        return training_state, env_state

    for i in range(100):
        print(params['params']['hidden_0']['kernel'].dtype)
        print(normalizer_params.mean.dtype)

        epoch_key, rng = jax.random.split(rng)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        training_state, env_state = training_epoch_with_timing(training_state, env_state, epoch_keys)
