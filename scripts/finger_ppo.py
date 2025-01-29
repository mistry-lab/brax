from jax import config
import os
import argparse

import wandb
from tensorboardX import SummaryWriter

from datetime import datetime
import functools

from brax.training.agents.ppo import train as ppo
from brax.envs.fd.finger import Finger

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def create_wandb_run(alg_name, env_name, seed, notes, run_id=None):

    name = f"{alg_name}_{env_name}_sweep_{seed}"

    return wandb.init(
        project="Brax",
        group="analytic_gradients",
        entity="alexanderwebb03-university-of-edinburgh",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )

def progress(times, writer, num_steps, metrics):
    times.append(datetime.now())
    print(num_steps, metrics)

    log_scalar = lambda key, value: writer.add_scalar(f"{key}", value, num_steps)
    for reward in ["reward"]:
      log_scalar(f"eval/episode_{reward}", metrics[f"eval/episode_{reward}"].item())
      log_scalar(f"eval/episode_{reward}_std", metrics[f"eval/episode_{reward}_std"].item())
    log_scalar(f"eval/avg_episode_length", metrics[f"eval/avg_episode_length"].item())
    log_scalar(f"eval/epoch_eval_time", metrics[f"eval/epoch_eval_time"])
    log_scalar(f"eval/sps", metrics[f"eval/sps"])

    writer.flush()

def main():
    env = Finger()

    log_dir = os.path.join("log")
    writer = SummaryWriter(log_dir)

    times = [datetime.now()]
    
    seed = 32
    create_wandb_run("shac2", "finger", seed, "test run")

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
        seed=seed
    )

if __name__=='__main__':
    main()
