import jax
import os

import wandb
from tensorboardX import SummaryWriter

from datetime import datetime
import functools

from brax.training.agents.ppo import train as ppo
from brax import envs

def progress(times, writer, num_steps, metrics):
    times.append(datetime.now())
    print(num_steps, metrics)

    # log_scalar = lambda key, value: writer.add_scalar(f"{key}", value, num_steps)
    # for reward in ["reward", "forward_reward", "reward_contact", "reward_survive", "reward_forward", "reward_ctrl", "x_velocity", "y_velocity"]:
    #   log_scalar(f"eval/episode_{reward}", metrics[f"eval/episode_{reward}"].item())
    #   log_scalar(f"eval/episode_{reward}_std", metrics[f"eval/episode_{reward}_std"].item())

    writer.flush()

def main():
    env = envs.get_environment(env_name="ant", backend="positional")
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    log_dir = os.path.join("log")
    writer = SummaryWriter(log_dir)

    times = [datetime.now()]
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        progress_fn=functools.partial(progress, times, writer),
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=16,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=512,
        seed=3
    )

if __name__=='__main__':
    main()
