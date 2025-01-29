# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from brax import envs
from brax import base
from brax.training import types, acting
from brax.training.types import Params, PRNGKey
from brax.training.agents.diffrl_shac import networks as shac_networks
from brax.training.acme import running_statistics, specs
from brax.training import gradients, replay_buffers
from brax.envs import wrappers

from brax.training.agents.diffrl_shac import losses as shac_losses

from absl import logging

import functools
from typing import Callable, Any

import jax.numpy as jnp
import optax
import jax
import flax

import time
import copy
from typing import Optional, List, Tuple

from shac.utils.common import *
from shac.utils.running_mean_std import RunningMeanStd
from shac.utils.time_report import TimeReport
from shac.utils.average_meter import AverageMeter

Metrics = types.Metrics
Transition = types.Transition

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: Params
  value_optimizer_state: optax.OptState
  value_params: Params
  target_value_params: Params
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray

def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

def train(
    environment: envs.Env,
    steps_num: int,  # horizon for short rollouts
    max_epochs: int,  # number of short rollouts to do (i.e. epochs)
    logdir: str,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    name: str = "shac",
    normalize_observations: bool = False,
    reward_scaling: float = 1.,
    grad_norm: Optional[float] = None,  # clip actor and ciritc grad norms
    critic_grad_norm: Optional[float] = None,
    actor_lr: float = 2e-3,
    critic_lr: float = 2e-3,
    betas: Tuple[float, float] = (0.7, 0.95),
    lr_schedule: str = "linear",
    gamma: float = 0.99,
    seed: int = 0,
    lam: float = 0.95,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    rew_scale: float = 1.0,
    obs_rms: bool = False,
    ret_rms: bool = False,
    critic_iterations: int = 16,
    critic_batches: int = 4,
    critic_method: str = "one-step",
    target_critic_alpha: float = 0.4,
    score_keys: List[str] = [],
    eval_runs: int = 12,
    log_jacobians: bool = False,  # expensive and messes up wandb
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
              shac_networks.DiffRLSHACNetworks] = shac_networks.make_shac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None
):
    # sanity check parameters
    assert steps_num > 0
    assert max_epochs >= 0
    assert actor_lr >= 0
    assert critic_lr >= 0
    assert lr_schedule in ["linear", "constant"]
    assert 0 < gamma <= 1
    assert 0 < lam <= 1
    assert rew_scale > 0.0
    assert critic_iterations > 0
    assert critic_batches > 0
    assert critic_method in ["one-step", "td-lambda"]
    assert 0 < target_critic_alpha <= 1.0
    assert eval_runs >= 0

    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    elif isinstance(environment, envs.Env):
        wrap_for_training = envs.training.wrap

    env = wrap_for_training(
        environment,
        episode_length=steps_num,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )  # pytype: disable=wrong-keyword-args

    num_timesteps = num_envs * steps_num * max_epochs
    if min_replay_size >= num_timesteps:
        raise ValueError(
            'No training will happen because min_replay_size >= num_timesteps'
        )

    if max_replay_size is None:
        max_replay_size = num_timesteps

    reset_fn = jax.jit(jax.vmap(env.reset))

    # Create environment
    print("num_envs = ", num_envs)

    critic_batch_size = num_envs * steps_num // critic_batches

    ret_rms = None
    if ret_rms:
        ret_rms = RunningMeanStd(shape=())

    env_name = environment.__class__.__name__
    name = name + "_" + env_name

    # Create actor and critic
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    shac_network = network_factory(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=normalize)
    make_policy = shac_networks.make_inference_fn(shac_network)

    critic = shac_network.value_network
    target_critic = copy.deepcopy(critic)

    # initialize optimizers
    value_optimizer = optax.chain(
        None if critic_grad_norm is None else optax.clip(critic_grad_norm),
        optax.adam(learning_rate=critic_lr, b1=betas[0], b2=betas[1])
    )

    value_gradient_update_fn = gradients.gradient_update_fn(
      shac_losses.compute_critic_loss, value_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    policy_optimizer = optax.chain(
        None if grad_norm is None else optax.clip(grad_norm),
        optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
    )

    jacs = []
    cfs = []
    body_fs = []
    accs = []
    early_terms = []
    horizon_truncs = []
    episode_ends = []
    episode = 0

    policy_loss_fn = functools.partial(
        shac_losses.compute_actor_loss,
        steps_num,
        num_envs,
        ret_rms,
        shac_network
    )

    rollout_policy_loss_fn = functools.partial(
        rollout_loss_fn,
        env,
        steps_num,
        reward_scaling,
        make_policy,
        policy_loss_fn
    )

    policy_gradient_update_fn = gradients.gradient_update_fn(
      rollout_policy_loss_fn, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    policy_gradient_update_fn = jax.jit(policy_gradient_update_fn)

    # counting variables
    iter_count = 0
    step_count = 0

    # loss variables
    episode_length_his = []
    episode_loss_his = []
    episode_discounted_loss_his = []

    best_policy_loss = np.inf
    actor_loss = np.inf
    value_loss = np.inf
    early_termination = 0
    episode_end = 0
    log_jacobians = log_jacobians
    eval_runs = eval_runs
    last_log_steps = 0

    # average meter
    episode_loss_meter = AverageMeter(1, 100)
    episode_discounted_loss_meter = AverageMeter(1, 100)
    episode_length_meter = AverageMeter(1, 100)
    horizon_length_meter = AverageMeter(1, 100)
    score_keys = score_keys
    episode_scores_meter_map = {
        key + "_final": AverageMeter(1, 100)
        for key in score_keys
    }

    # timer
    time_report = TimeReport()

    start_time = time.time()

    # add timers
    time_report.add_timer("algorithm")
    time_report.add_timer("compute actor loss")
    time_report.add_timer("forward simulation")
    time_report.add_timer("backward simulation")
    time_report.add_timer("prepare critic dataset")
    time_report.add_timer("actor training")
    time_report.add_timer("critic training")

    time_report.start_timer("algorithm")

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)

    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d', jax.device_count(), process_count,
        process_id, local_device_count, local_devices_to_use)
    device_count = local_devices_to_use * process_count
    assert num_envs % device_count == 0

    replay_buffer = \
        initialize_replay_buffer(
            environment.observation_size,
            environment.action_size,
            max_replay_size,
            device_count, 
            critic_batch_size
    )

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, rb_key, eval_key = jax.random.split(local_key, 4)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs,
                            (local_devices_to_use, -1) + key_envs.shape[1:])

    policy_init_params = shac_network.policy_network.init(key_policy)
    value_init_params = shac_network.value_network.init(key_value)

    training_state = TrainingState(
      policy_optimizer_state=policy_optimizer.init(policy_init_params),
      policy_params=policy_init_params,
      value_optimizer_state=value_optimizer.init(value_init_params),
      value_params=value_init_params,
      target_value_params=value_init_params,
      normalizer_params=running_statistics.init_state(
          specs.Array((env.observation_size,), jnp.float32)),
      env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])

    # initializations
    env_state = reset_fn(key_envs)

    if not eval_env:
        eval_env = env
    else:
        eval_env = wrappers.wrap_for_training(
            eval_env, episode_length=steps_num, action_repeat=action_repeat)

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=steps_num,
        action_repeat=action_repeat,
        key=eval_key)

    # Run initial eval
    if process_id == 0:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (training_state.normalizer_params, training_state.policy_params)),
            training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics)


    pmap_training_epoch = jax.pmap(
        functools.partial(training_epoch,
            num_envs,
            critic_method,
            replay_buffer,
            gamma,
            steps_num,
            critic_iterations,
            critic_batch_size,
            time_report,
            value_gradient_update_fn,
            policy_gradient_update_fn,
            shac_network,
            reward_scaling),
        axis_name=_PMAP_AXIS_NAME
    )

    # main training process
    for epoch in range(max_epochs):
        time_start_epoch = time.time()
        
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        metrics = pmap_training_epoch(
            buffer_state,
            epoch_keys,
            env_state,
            training_state
        )
        epoch_metrics(
            metrics,
            episode_loss_his,
            episode_length_meter,
            episode_loss_meter,
            episode_discounted_loss_meter,
            score_keys,
            episode_scores_meter_map
        )

        if (
            log_jacobians
            and step_count - last_log_steps > 1000 * num_envs
        ):
            np.savez(
                os.path.join(logdir, f"truncation_analysis_{episode}"),
                contact_forces=torch.cat(cfs).numpy(),
                body_forces=torch.cat(body_fs).numpy(),
                accelerations=torch.cat(accs).numpy(),
                early_termination=early_terms,
                horizon_truncation=horizon_truncs,
                episode_ends=episode_ends,
            )
            cfs = []
            body_fs = []
            accs = []
            early_terms = []
            horizon_truncs = []
            episode_ends = []
            episode += 1
            last_log_steps = step_count

        update_target_critic(target_critic_alpha, critic, target_critic)

        time_end_epoch = time.time()
        fps = steps_num * num_envs / (time_end_epoch - time_start_epoch)

        progress_fn(epoch, metrics)

    time_report.end_timer("algorithm")

    time_report.report()

    # save reward/length history
    np.save(
        open(os.path.join(logdir, "episode_loss_his.npy"), "wb"),
        episode_loss_his,
    )
    np.save(
        open(os.path.join(logdir, "episode_discounted_loss_his.npy"), "wb"),
        episode_discounted_loss_his,
    )
    np.save(
        open(os.path.join(logdir, "episode_length_his.npy"), "wb"),
        episode_length_his,
    )

def initialize_replay_buffer(
        observation_size: int,
        action_size: int,
        max_replay_size: int,
        device_count: int,
        critic_batch_size: int):

    dummy_obs = jnp.zeros((observation_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
    )

    return replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=critic_batch_size // device_count,
    )

def rollout_loss_fn(
        env: envs.Env,
        unroll_length: int,
        reward_scaling: float,
        make_policy: types.Policy,
        policy_loss_fn,
        policy_params,
        value_params,
        normalizer_params,
        state, 
        key
    ):

    policy = make_policy((normalizer_params, policy_params))

    key, key_loss = jax.random.split(key)

    next_state, data = acting.generate_unroll(
        env,
        state,
        policy,
        key,
        unroll_length,
        extra_fields=('truncation',))

    data = Transition(
        observation=data.observation,
        action=data.action,
        reward=reward_scaling * data.reward,
        discount=data.discount,
        next_observation=data.next_observation,
        extras=data.extras
    )

    loss, metrics = policy_loss_fn(policy_params, value_params,
        normalizer_params, data, key_loss)

    return loss, (state, data, metrics)

def epoch_metrics(
        metrics,
        episode_loss_his,
        episode_length_meter,
        episode_loss_meter,
        episode_discounted_loss_meter,
        score_keys,
        episode_scores_meter_map,  
    ):

    if len(episode_loss_his) > 0:
        mean_episode_length = episode_length_meter.get_mean()
        mean_policy_loss = episode_loss_meter.get_mean()
        mean_policy_discounted_loss = (
            episode_discounted_loss_meter.get_mean()
        )

        if (
            score_keys
            and len(
                episode_scores_meter_map[score_keys[0] + "_final"]
            )
            > 0
        ):
            for score_key in score_keys:
                score = episode_scores_meter_map[
                    score_key + "_final"
                ].get_mean()
                metrics[f"scores/{score_key}"] = score

        metrics.update({
            "policy_loss": mean_policy_loss,
            "rewards": -mean_policy_loss,
            "policy_discounted_loss": mean_policy_discounted_loss,
            "best_policy_loss": best_policy_loss,
            "episode_lengths": mean_episode_length,
            "ac_std": actor.get_logstd().exp().mean().detach().cpu().item(),
            "actor_grad_norm": grad_norm_before_clip,
            "episode_end": episode_end,
            "early_termination": early_termination
        })

    else:
        mean_policy_loss = np.inf
        mean_policy_discounted_loss = np.inf
        mean_episode_length = 0

    print(
        "iter {:}/{:}, ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, avg rollout {:.1f}, total steps {:}, fps {:.2f}, value loss {:.2f}, grad norm before/after clip {:.2f}/{:.2f}".format(
            iter_count,
            max_epochs,
            mean_policy_loss,
            mean_policy_discounted_loss,
            mean_episode_length,
            horizon_length_meter.get_mean(),
            step_count,
            fps,
            value_loss,
            grad_norm_before_clip,
            grad_norm_after_clip,
        )
    )

def update_target_critic(target_critic_alpha, critic, target_critic):
    with torch.no_grad():
        alpha = target_critic_alpha
        for param, param_targ in zip(
            critic.parameters(), target_critic.parameters()
        ):
            param_targ.data.mul_(alpha)
            param_targ.data.add_((1.0 - alpha) * param.data)

def training_epoch(
    num_envs: int,
    critic_method: str,
    replay_buffer: replay_buffers.ReplayBuffer,
    gamma: float,
    steps_num,
    critic_iterations: int,
    critic_batch_size: int,
    time_report,
    value_gradient_update_fn,
    policy_gradient_update_fn,
    shac_network,
    reward_scaling,
    buffer_state: ReplayBufferState,
    key: PRNGKey,
    state: envs.State,
    training_state):

    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    # train actor
    time_report.start_timer("actor training")
    (policy_loss, (state, data, policy_metrics)), policy_params, policy_optimizer_state = policy_gradient_update_fn(
        training_state.target_value_params, training_state.normalizer_params, state, key_generate_unroll,
        optimizer_state=training_state.policy_optimizer_state)
    time_report.end_timer("actor training")

    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        data.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)

    # train critic
    # prepare dataset
    time_report.start_timer("prepare critic dataset")
    with torch.no_grad():
        target_values = shac_losses.compute_target_values(
            num_envs,
            critic_method,
            shac_network,
            training_state.target_value_params,
            normalizer_params,
            data,
            gamma,
            steps_num,
            reward_scaling
        )

        buffer_state = replay_buffer.insert(
            buffer_state,
            {
                "observation": data.observation,
                "target_values": target_values
            }
        )

    time_report.end_timer("prepare critic dataset")

    time_report.start_timer("critic training")
    value_loss = 0.0
    buffer_state, transitions = replay_buffer.sample(buffer_state)

    for j in range(critic_iterations):
        value_optimizer_state, value_params, key_sgd \
            = critic_sgd_step(training_state.value_optimizer_state, training_state.value_params, key_sgd, transitions, value_gradient_update_fn, training_state.normalizer_params, data)

    time_report.end_timer("critic training")

    iter_count += 1

    # logging
    metrics = {
        "lr": lr,
        "actor_loss": policy_loss,
        "value_loss": value_loss,
        "rollout_len": horizon_length_meter.get_mean(),
        "fps": fps
    }


def critic_sgd_step(
        optimizer_state,
        params,
        key: PRNGKey,
        transitions,
        value_gradient_update_fn,
        normalizer_params,
        data
    ):

    total_critic_loss = 0.0
    batch_cnt = 0

    # critic_loss, q_params, q_optimizer_state = critic_update(
    #     training_state.q_params,
    #     training_state.policy_params,
    #     training_state.normalizer_params,
    #     training_state.target_q_params,
    #     alpha,
    #     transitions,
    #     key_critic,
    #     optimizer_state=training_state.q_optimizer_state,
    # )

    for i in range(len(dataset)):
        batch_sample = dataset[i]

        key, key_loss = jax.random.split(key)
        (training_critic_loss, metrics), params, optimizer_state = value_gradient_update_fn(
            params,
            normalizer_params,
            data,
            optimizer_state=optimizer_state)

        # ugly fix for simulation nan problem
        # for params in critic.parameters():
        #     params.grad.nan_to_num_(0.0, 0.0, 0.0)

        # if critic_grad_norm:
        #     clip_grad_norm_(critic.parameters(), critic_grad_norm)

        total_critic_loss += training_critic_loss
        batch_cnt += 1

    value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
    print(
        "value iter {}/{}, loss = {:7.6f}".format(
            j + 1, critic_iterations, value_loss
        ),
        end="\r",
    )

    return optimizer_state, params, key
