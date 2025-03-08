from typing import List, Optional, Tuple
import optax

def get_optimizer(
        schedule: str,
        learning_rate: float,
        grad_norm: float,
        betas: List[float],
        number_steps: Optional[int] = None
) -> Tuple[optax.GradientTransformationExtraArgs, Optional[optax.Schedule]]:
    if grad_norm is None and schedule == "constant":
        return optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1]), None
    elif schedule == "constant":
        return optax.chain(
            optax.clip_by_global_norm(grad_norm),
            optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1]),
        ), None

    elif grad_norm is None and schedule == "linear":
        schedule = optax.linear_schedule(learning_rate, 1e-5, number_steps)

        return optax.chain(
            optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1]),
            optax.scale_by_schedule(schedule)
        ), schedule

    schedule = optax.linear_schedule(learning_rate, 1e-5, number_steps)

    return optax.chain(
        optax.clip_by_global_norm(grad_norm),
        optax.adam(learning_rate=learning_rate, b1=betas[0], b2=betas[1]),
        optax.scale_by_schedule(schedule),
    ), schedule
