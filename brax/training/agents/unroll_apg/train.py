from brax import envs
from brax import base
from brax.training import types, acting
from brax.training import pmap
from brax.training import logger as metric_logger
from brax.training.types import Params, PRNGKey
from brax.training.acme import running_statistics, specs
from brax.training import gradients
from brax.v1 import envs as envs_v1

from brax.training.agents.unroll_apg import checkpoint
from brax.training.agents.unroll_apg import losses as apg_losses
from brax.training.agents.unroll_apg import networks as apg_networks

from absl import logging

import functools
from typing import Callable, Any, Union, Dict

import numpy as np
import jax.numpy as jnp
import optax
import jax
import flax

import time
from typing import Optional, Tuple

Metrics = types.Metrics
Transition = types.Transition

ReplayBufferState = Any

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

def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)

def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    apg_network: apg_networks.UnrollAPGNetworks,
    policy_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    params = apg_network.policy_network.init(key)
    optimizer_state = policy_optimizer.init(params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float32'))
    )

    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=jnp.zeros(()),
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

def _maybe_wrap_env(
    env: Union[envs_v1.Env, envs.Env],
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified in ppo.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    elif isinstance(env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env

def train(
    environment: envs.Env,
    # schedule parameters
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    batch_size: int = 32,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # network parameters
    network_factory: types.NetworkFactory[
              apg_networks.UnrollAPGNetworks] = apg_networks.make_apg_network,
    grad_norm: Optional[float] = None,  # clip actor and ciritc grad norms
    lr_schedule: str = "linear",
    learning_rate: float = 2e-3,
    betas: Tuple[float, float] = (0.7, 0.95),
    # SHAC params
    discounting: float = 0.9,
    normalize_observations: bool = False,
    reward_scaling: float = 1.,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_value_fn: bool = True,
):
    # sanity check parameters
    assert batch_size  % num_envs == 0

    assert num_timesteps >= 0
    assert learning_rate >= 0
    assert lr_schedule in ["linear", "constant"]
    assert 0 < discounting <= 1
    assert reward_scaling > 0.0

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)

    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    assert num_envs % device_count == 0

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        action_repeat * batch_size * episode_length
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
        )
    ).astype(int)

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        local_device_count,
        key,
        wrap_env_fn,
        randomization_fn,
    )

    obs_size = env.observation_size
    if isinstance(obs_size, Dict):
        raise NotImplementedError('Dictionary observations not implemented in SHAC')

    # Create actor and critic
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    apg_network = network_factory(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=normalize)
    make_policy = apg_networks.make_inference_fn(apg_network)

    # initialize optimizers
    policy_optimizer = optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1])
    if grad_norm is not None:
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(grad_norm),
            optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1])
        )

    loss_fn = functools.partial(apg_losses.compute_apg_loss,
        apg_network=apg_network,
        env=env,
        episode_length=episode_length,
        number=batch_size // num_envs,
        discounting=discounting,
        reward_scaling=reward_scaling,
    )

    actor_update = gradients.gradient_update_fn(
      loss_fn, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_metrics_steps
        or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    ckpt_config = checkpoint.network_config(
        observation_size=obs_size,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
    )

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState,
        Union[envs.State, envs_v1.State],
        Metrics,
    ]:
        # train actor
        (_, extras), params, optimizer_state = actor_update(
            training_state.params,
            training_state.normalizer_params,
            env_state,
            key,
            optimizer_state=training_state.optimizer_state)

        data = extras["data"]
        next_state = extras["next_state"]

        if log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras['state_extras']['episode_metrics'],
                data.extras['state_extras']['episode_done'],
            )

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
            env_steps=training_state.env_steps + env_step_per_training_step,
        )

        return training_state, next_state, {} # metrics

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

        def f(carry, unused_t):
            ts, es, k = carry
            k, new_key = jax.random.split(k)
            ts, es, metrics = training_step(ts, es, k)
            return (ts, es, new_key), metrics

        (training_state, env_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        # training_state, env_state = _strip_weak_type((training_state, env_state))
        (training_state, env_state, metrics) = training_epoch(
            training_state, env_state, key
        )
        # training_state, env_state, metrics = \
        #     _strip_weak_type((training_state, env_state, metrics))

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            env_step_per_training_step * num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    rng = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        apg_network=apg_network,
        policy_optimizer=policy_optimizer,
    )
    del global_key

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else training_state.value_training_state.params
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(
                policy=params[1], value=value_params
            ),
        )

    local_key, env_key, eval_key = jax.random.split(local_key, 3)

    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(
        env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
    )
    env_state = jax.pmap(env.reset)(env_keys)

    eval_env = _maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        local_device_count=1,  # eval on the host only
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
    )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key)

    # Run initial eval
    metrics = {}
    if process_id == 0:
        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params,
            )),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, training_metrics) = (
            training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
        )

        if process_id != 0:
            continue

        current_step = int(_unpmap(training_state.env_steps))

        # Process id == 0.
        params = _unpmap((
            training_state.normalizer_params,
            training_state.params
        ))

        if save_checkpoint_path is not None:
            checkpoint.save(
                save_checkpoint_path, current_step, params, ckpt_config
            )

        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
