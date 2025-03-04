# Copyright 2024 The Brax Authors.
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

"""Brax training gradient utility functions."""

from typing import Callable, Optional

import jax
import optax

import flax
from brax.training.types import Params

@flax.struct.dataclass
class NetworkTrainingState:
    optimizer_state: optax.OptState
    params: Params

def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux, argnums=(0, 1))

    def h(*args, **kwargs):
        value, grads = g(*args, **kwargs)
        return value, \
            jax.lax.pmean(grads[0], axis_name=pmap_axis_name), \
            jax.lax.pmean(grads[1], axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def dual_gradient_update_fn(
    loss_fn: Callable[..., float],
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
        loss_fn: The loss function.
        optimizer: The optimizer to apply gradients.
        pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
        has_aux: Whether the loss_fn has auxiliary data.

    Returns:
        A function that takes the same argument as the loss function plus the
        optimizer state. The output of this function is the loss, the new parameter,
        and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(*args, policy_optimizer_state, value_optimizer_state):
        value, policy_grads, value_grads = loss_and_pgrad_fn(*args)
        policy_params_update, policy_optimizer_state = policy_optimizer.update(policy_grads, policy_optimizer_state)
        policy_params = optax.apply_updates(args[0], policy_params_update)

        value_params_update, value_optimizer_state = value_optimizer.update(value_grads, value_optimizer_state)
        value_params = optax.apply_updates(args[1], value_params_update)

        return value, \
            NetworkTrainingState(
                optimizer_state=policy_optimizer_state,
                params=policy_params,
            ), \
            NetworkTrainingState(
                optimizer_state=value_optimizer_state,
                params=value_params,
            )

    return f
