import importlib
import functools
from omegaconf import DictConfig, OmegaConf

from typing import Optional, Callable, Tuple, Union, Any

import hydra

from brax import envs
from brax import base
from brax.training.agents.diffrl_shac import networks as shac_networks
from brax.training.agents.diffrl_shac.unroll import generate_batched_unroll, generate_unroll as single_unroll
from brax.training.agents.diffrl_shac.losses import compute_td_value

from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.envs.fd import get_environment
from brax.training.types import PRNGKey, Params
from brax.v1 import envs as envs_v1

import optax
import jax
import flax
from jax import config
import jax.numpy as jnp

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

_PMAP_AXIS_NAME = 'i'

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_params: Params
    policy_optimizer_state: optax.OptState
    value_params: Params
    normalizer_params: running_statistics.RunningStatisticsState

def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  loss_and_pgrad_fn = loss_and_pgrad(
    loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state

  return f

def unroll(
    environment: envs.Env,
    network_factory: types.NetworkFactory[
              shac_networks.DiffRLSHACNetworks] = shac_networks.make_shac_networks,
    reward_scaling: float = 1.,
    actor_grad_norm: Optional[float] = 0.5,
    actor_lr: float = 2e-3,
    unroll_length: int = 10,
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
    rng, key_value, key_policy, unroll_key, env_key = jax.random.split(rng, 5)

    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count

    print(num_envs)

    env = envs.training.wrap(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )

    normalize = lambda x, y: x
    # if normalize_observations:
    #     normalize = running_statistics.normalize

    shac_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize)

    env_keys = jax.random.split(env_key, 4)
    # env_keys = jnp.reshape(
    #     env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
    # )
    # env_state = jax.pmap(env.reset)(env_keys)
    env_state = env.reset(env_keys)

    obs_size = env.observation_size
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float32'))
    )

    policy_params = shac_network.policy_network.init(key_policy)
    make_policy = shac_networks.make_inference_fn(shac_network)

    value_params = shac_network.value_network.init(key_value)
    value_apply = shac_network.value_network.apply

    policy_optimizer = optax.adam(learning_rate=actor_lr)
    if actor_grad_norm is not None:
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(actor_grad_norm),
            optax.adam(learning_rate=actor_lr)
        )

    def actor_loss(
        policy_params: Params,
        value_params: Params,
        normalizer_params: Any,
        env: envs.Env,
        env_state: envs.State
    ):
        policy = make_policy((normalizer_params, policy_params))

        final_state, data = single_unroll(
            env=env,
            env_state=env_state,
            policy=policy,
            reward_scaling=2.0,
            unroll_length=8,
            extra_fields=('truncation', 'episode_metrics', 'episode_done'),
        )
        # Put the time dimension first.
        ordered_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        truncation = ordered_data.extras['state_extras']['truncation']

        values = value_apply(normalizer_params, value_params, ordered_data.observation)
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], ordered_data.next_observation)
        bootstrap_value = value_apply(normalizer_params, value_params, terminal_obs)

        termination = (1 - ordered_data.discount) * (1 - truncation)

        # value = compute_td_value(
        #     truncation=truncation,
        #     termination=termination,
        #     rewards=ordered_data.reward,
        #     values=values,
        #     bootstrap_value=bootstrap_value,
        #     discount=1.
        # )

        return jnp.mean(ordered_data.reward[-1]), {
            "final_state": final_state,
            "data": data
        }

    unroll_fn = gradient_update_fn(
        actor_loss,
        policy_optimizer,
        pmap_axis_name=None,
        has_aux=True
    )

    def generate_unroll(
        training_state: TrainingState,
        state
    ):
        (loss, extras), policy_params, policy_optimizer_state = unroll_fn(
            training_state.policy_params,
            training_state.value_params,
            training_state.normalizer_params,
            env,
            state,
            optimizer_state=training_state.policy_optimizer_state,
        )

        data = extras["data"]

        # normalizer_params = running_statistics.update(
        #     training_state.normalizer_params,
        #     data.observation,
        #     pmap_axis_name=_PMAP_AXIS_NAME,
        # )

        training_state = TrainingState(
            policy_params=policy_params,
            policy_optimizer_state=policy_optimizer_state,
            value_params=training_state.value_params,
            normalizer_params=training_state.normalizer_params
        )

        return training_state, loss, extras

    # generate_unroll = jax.pmap(generate_unroll, axis_name=_PMAP_AXIS_NAME)
    
    policy_optimizer_state = policy_optimizer.init(policy_params)
    training_state = TrainingState(
        policy_params=policy_params,
        policy_optimizer_state=policy_optimizer_state,
        value_params=value_params,
        normalizer_params=normalizer_params
    )

    # training_state = jax.device_put_replicated(
    #     training_state, jax.local_devices()[:local_devices_to_use]
    # )

    for i in range(100):
        training_state, value_loss, extras = generate_unroll(training_state, env_state)
        env_state = extras["final_state"]

        print(value_loss)

        metrics = jax.tree_util.tree_map(jnp.mean, value_loss)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        print(metrics)


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    alg_module = importlib.import_module("brax.training.agents.diffrl_shac.single_step")

    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    param_subset = {}
    for name in [
        "reward_scaling",
        "unroll_length",
        "batch_size",
        "num_envs",
        "normalize_observations",
        "episode_length",
        "action_repeat",
        "randomization_fn"
    ]:
        if name in cfg_full["alg"]["params"]:
            param_subset[name] = cfg_full["alg"]["params"][name]

    env = get_environment(cfg.env.name)

    unroll(
        environment=env,
        **param_subset
    )

if __name__ == '__main__':
    main()
