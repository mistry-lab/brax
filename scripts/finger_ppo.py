import os
import importlib
import functools
import hydra
import pathlib
import logging
import argparse
from hydra.core.hydra_config import HydraConfig
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import List, Callable

import wandb
from tensorboardX import SummaryWriter

from jax import config
from brax.envs import Env, get_environment
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper, AutoResetWrapper, VmapWrapper
from brax.envs.fd import get_environment as get_fd_environment
from brax.envs.fd.wrappers import get_terminal_reward_wrapper

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def create_wandb_run(
        project: str,
        group: str,
        entity: str,
        alg_name: str,
        env_name: str,
        seed: int,
        notes: str | None,
        job_config: dict, 
        run_id: str,
        resume: bool = False
    ):

    name = f"{alg_name}_{env_name}_sweep_{seed}"

    return wandb.init(
        project=project,
        config=job_config,
        group=group,
        entity=entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=resume,
    )

def get_time_from_path(path: str):
    parts = pathlib.Path(path).parts

    date = parts[-2]
    time = parts[-1]

    return f"{date}T{time}"

def progress(times: List[datetime], writer: SummaryWriter, num_steps: int, metrics: dict):
    times.append(datetime.now())

    log_scalar = lambda key, value: writer.add_scalar(f"{key}", value, num_steps)
    for reward in ["reward"]:
      log_scalar(f"eval/episode_{reward}", metrics[f"eval/episode_{reward}"].item())
      log_scalar(f"eval/episode_{reward}_std", metrics[f"eval/episode_{reward}_std"].item())
    log_scalar(f"eval/avg_episode_length", metrics[f"eval/avg_episode_length"].item())
    log_scalar(f"eval/epoch_eval_time", metrics[f"eval/epoch_eval_time"])
    log_scalar(f"eval/sps", metrics[f"eval/sps"])

    writer.flush()

def wrap_env_fn(
    env: Env,
    episode_length: int,
    action_repeat: int,
    randomization_fn: Callable,
    terminal_reward_name: str, 
):
    logging.info(f"Wrapping with terminal reward function: {terminal_reward_name}")
    
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)

    env = get_terminal_reward_wrapper(
        env=env,
        terminal_reward_name=terminal_reward_name,
        episode_length=episode_length,
        action_repeat=action_repeat,
        vmap=True
    )
    env = AutoResetWrapper(env)

    return env

def get_command_line_args():
    parser = argparse.ArgumentParser(description='Script to train brax models.')    
    parser.add_argument('-r', '--regular', action='store_true', help='Use regular instead of FD environment.')

    args = parser.parse_args()
    return args

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    hydra_logdir = HydraConfig.get()["runtime"]["output_dir"]

    param_copy_dir = os.path.join(hydra_logdir, "resolved_config.yaml")
    OmegaConf.save(cfg, param_copy_dir, resolve=True)

    if cfg.general.debug:
        cfg.alg.params.num_evals = 0
        cfg.alg.params.num_minibatches = 1
        cfg.alg.params.num_envs = 1
        cfg.alg.params.batch_size = 1

        os.environ["CUDA_VISIBLE_DEVICES"]="3"

    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if not cfg.general.debug and not cfg.general.no_wandb:
        create_wandb_run(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            entity=cfg.wandb.entity,
            alg_name=cfg.alg.name,
            env_name=cfg.env.name,
            seed=cfg.general.seed,
            notes=cfg.wandb.notes,
            job_config=cfg_full,
            run_id=get_time_from_path(hydra_logdir),
            resume=cfg.wandb.resume
        )

    logdir = os.path.join(hydra_logdir, cfg.general.logdir)
    writer = SummaryWriter(logdir)

    times = [datetime.now()]

    alg_module = importlib.import_module(cfg.alg.module_path)
    algorithm_train = getattr(alg_module, "train")

    param_dir = os.path.join(hydra_logdir, cfg.general.param_dir)
    os.makedirs(param_dir)

    kwrd_args = {cfg.alg.checkpoint_keyword_arg: param_dir}
    if "network_factory_path" in cfg.alg:
        network_factory_path = importlib.import_module(cfg.alg.network_factory_path)
        network_factory = getattr(network_factory_path, cfg.alg.make_network_fn_name)
        network_kwargs = {
            key: OmegaConf.to_object(cfg.alg.network_factory_params[key]) \
                for key in cfg.alg.network_factory_params}
        network_factory = functools.partial(
            network_factory,
            **network_kwargs
        )
        kwrd_args["network_factory"] = network_factory

    args = get_command_line_args()
    env = get_environment(cfg.env.name) if args.regular else get_fd_environment(cfg.env.name)
    make_inference_fn, params, _ = algorithm_train(
        **cfg.alg.params,
        environment=env,
        wrap_env_fn=functools.partial(wrap_env_fn, terminal_reward_name=cfg.env.terminal_reward_name) \
            if "terminal_reward_name" in cfg.env else None,
        progress_fn=functools.partial(progress, times, writer),
        seed=cfg.general.seed,
        **kwrd_args
    )

    if not cfg.general.debug and not cfg.general.no_wandb:
        wandb.finish()

if __name__=='__main__':
    train()
