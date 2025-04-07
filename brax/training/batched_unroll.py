from typing import Tuple

import jax
import jax.numpy as jnp

from typing import Tuple

from brax import envs
import jax

from brax import envs

from brax.training import types
from brax.training.acting import generate_unroll
from brax.training.types import PRNGKey

def generate_batched_unroll(
        env: envs.Env,
        env_state: envs.State,
        policy: types.Policy,
        key: PRNGKey,
        unroll_length: int,
        number: int,
        reward_scaling: float,
        extra_fields: Tuple[str] = (),
        include_time: bool = False,
        **kwargs
    ):

    def f(carry, unused_t):
        current_state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        next_state, data = generate_unroll(
            env=env,
            env_state=current_state,
            policy=policy,
            key=key,
            unroll_length=unroll_length,
            extra_fields=extra_fields,
            include_time=include_time,
            **kwargs
        )

        data = types.Transition(
            observation=data.observation,
            action=data.action,
            reward=reward_scaling * data.reward,
            discount=data.discount,
            next_observation=data.next_observation,
            extras=data.extras
        )

        return (next_state, next_key), data

    (final_state, _), data = jax.lax.scan(
        f,
        (env_state, key),
        (),
        length=number,
    )
    # Have leading dimensions (batch_size, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (unroll_length,)

    return final_state, data
