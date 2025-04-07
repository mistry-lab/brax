from typing import Optional, Callable, Tuple

from brax import envs
from brax import base
from brax.training.agents.diffrl_shac import networks as shac_networks
from brax.training.agents.diffrl_shac import losses as shac_losses
from brax.training.agents.diffrl_shac import checkpoint

from brax.training import types
from brax.training.types import Params
from brax.training import gradients
from brax.training.acme import running_statistics, specs

import jax
import flax
import jax.numpy as jnp

import optax

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

@flax.struct.dataclass
class NetworkTrainingState:
    optimizer_state: optax.OptState
    params: Params
    gradient_steps: jnp.ndarray

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_training_state: NetworkTrainingState
    value_training_state: NetworkTrainingState
    target_value_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray

def step(
    environment: envs.Env,
    include_time: bool = False,
    network_factory: types.NetworkFactory[
              shac_networks.DiffRLSHACNetworks] = shac_networks.make_shac_networks,
    discounting: float = 0.9,
    reward_scaling: float = 1.,
    gae_lambda: float = 0.95,
    actor_grad_norm: Optional[float] = None,
    value_grad_norm: Optional[float] = None,
    actor_lr: float = 2e-3,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 4,
    betas: Tuple[float, float] = (0.7, 0.95),
    deterministic_train: bool = False,
    num_envs: int = 1,
    normalize_observations: bool = False,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    save_checkpoint_path: Optional[str] = None,
):
    rng = jax.random.PRNGKey(1)
    rng, key_value, key_policy, env_key = jax.random.split(rng, 4)

    env = envs.training.wrap(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    shac_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize)

    critic_loss, actor_loss = shac_losses.make_losses(
        shac_network=shac_network,
        env=env,
        include_time=include_time,
        deterministic_train=deterministic_train,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        unroll_length=unroll_length,
        number=batch_size * num_minibatches // num_envs,
    )

    policy_optimizer = optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
    if actor_grad_norm is not None:
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(actor_grad_norm),
            optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
        )

    value_optimizer = optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
    if value_grad_norm is not None:
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(value_grad_norm),
            optax.adam(learning_rate=actor_lr, b1=betas[0], b2=betas[1])
        )

    update_fn = gradients.gradient_update_fn(
      actor_loss, policy_optimizer, pmap_axis_name=None, has_aux=True)

    obs_size = env.observation_size
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float64'))
    )


    policy_params = shac_network.policy_network.init(key_policy)
    value_params = shac_network.value_network.init(key_value)

    policy_optimizer_state = policy_optimizer.init(policy_params)
    value_optimizer_state = value_optimizer.init(value_params)

    env_keys = jax.random.split(env_key, num_envs)
    env_state = env.reset(env_keys)

    ckpt_config = checkpoint.network_config(
        observation_size=obs_size,
        action_size=env.action_size,
        normalize_observations=normalize_observations,
        network_factory=network_factory,
    )

    training_state = TrainingState(
        policy_training_state=NetworkTrainingState(
            optimizer_state=policy_optimizer_state,
            params=policy_params,
            gradient_steps=jnp.zeros(()),
        ),
        value_training_state=NetworkTrainingState(
            optimizer_state=value_optimizer_state,
            params=value_params,
            gradient_steps=jnp.zeros(()),
        ),
        target_value_params=value_params,
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params,
    )

    params = (
        training_state.normalizer_params,
        training_state.policy_training_state.params
    )

    checkpoint.save(
        save_checkpoint_path, 0, params, ckpt_config
    )
