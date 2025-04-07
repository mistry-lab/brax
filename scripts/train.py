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

def create_wandb_run(
        project: str,
        group: str,
        entity: str,
        alg_name: str,
        env_name: str,
        experiment_name: str,
        seed: int,
        job_config: dict, 
        run_id: str,
        resume: bool = False,
        **kwargs
    ):

    name = f"{alg_name}_{env_name}_{experiment_name}_{seed}"

    return wandb.init(
        project=project,
        config=job_config,
        group=group,
        entity=entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=[experiment_name],
        name=name,
        id=run_id,
        resume=resume,
        **kwargs
    )

def get_time_from_path(path: str):
    parts = pathlib.Path(path).parts

    date = parts[-2]
    time = parts[-1]

    return f"{date}T{time}"

def progress(times: List[datetime], writer: SummaryWriter, num_steps: int, metrics: dict):
    times.append(datetime.now())

    log_scalar = lambda key, value: writer.add_scalar(f"{key}", value, num_steps)
    for key, value in \
            ((key, value) for key, value in metrics.items() if \
                key.startswith("eval/episode") or key.endswith("_loss")):
        log_scalar(key, value.item())

    if "training/sps" in metrics:
        log_scalar(f"training/walltime", metrics[f"training/walltime"])
        log_scalar(f"training/sps", metrics[f"training/sps"])

    log_scalar(f"eval/walltime", metrics[f"eval/walltime"])
    log_scalar(f"eval/sps", metrics[f"eval/sps"])

    log_scalar(f"eval/avg_episode_length", metrics[f"eval/avg_episode_length"].item())
    log_scalar(f"eval/std_episode_length", metrics[f"eval/std_episode_length"].item())
    log_scalar(f"eval/epoch_eval_time", metrics[f"eval/epoch_eval_time"])

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

def convert_omegaconf(param):
    if OmegaConf.is_list(param):
        return OmegaConf.to_object(param)
    return param

def get_network_kwargs(alg_cfg):
    network_factory_path = importlib.import_module(alg_cfg.network_factory_path)
    network_factory = getattr(network_factory_path, alg_cfg.make_network_fn_name)
    network_kwargs = {
        key:  convert_omegaconf(alg_cfg.network_factory_params[key]) \
            for key in alg_cfg.network_factory_params}
    network_factory = functools.partial(
        network_factory,
        **network_kwargs
    )
    return {"network_factory": network_factory}

def train_with_cfg(cfg: DictConfig):
    hydra_logdir = HydraConfig.get()["runtime"]["output_dir"]

    if not cfg.general.debug and not cfg.general.no_wandb:    
        kwargs = {} if "sweep" in cfg.general else \
            {
                "notes": cfg.wandb.notes,
            }

        create_wandb_run(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            entity=cfg.wandb.entity,
            alg_name=cfg.alg.name,
            env_name=cfg.env.name,
            experiment_name=cfg.general.experiment_name,
            seed=cfg.general.seed,
            job_config=OmegaConf.to_container(cfg, resolve=True),
            run_id=get_time_from_path(hydra_logdir),
            resume=cfg.wandb.resume,
            **kwargs
        )

        cfg = OmegaConf.merge(cfg, dict(wandb.config))

    param_copy_dir = os.path.join(hydra_logdir, "resolved_config.yaml")
    OmegaConf.save(cfg, param_copy_dir, resolve=True)

    if cfg.general.debug:
        cfg.alg.params.num_evals = 0
        cfg.alg.params.num_minibatches = 1
        cfg.alg.params.num_envs = 1
        cfg.alg.params.batch_size = 1

        os.environ["CUDA_VISIBLE_DEVICES"]="3"

    logdir = os.path.join(hydra_logdir, cfg.general.logdir)
    writer = SummaryWriter(logdir)

    times = [datetime.now()]

    alg_module = importlib.import_module(cfg.alg.module_path)
    algorithm_train = getattr(alg_module, "train")

    param_dir = os.path.join(hydra_logdir, cfg.general.param_dir)
    os.makedirs(param_dir)

    kwargs = {cfg.alg.checkpoint_keyword_arg: param_dir}
    if "network_factory_path" in cfg.alg:
        if cfg.env.dtype == "float64":
            config.update('jax_default_matmul_precision', 'high')
            config.update("jax_enable_x64", True)

        kwargs.update(get_network_kwargs(cfg.alg))

    env_kwargs = OmegaConf.to_container(cfg.env.kwargs, resolve=True)
    env = get_environment(cfg.env.name, **env_kwargs) if cfg.general.regular_env \
        else get_fd_environment(cfg.env.name, **env_kwargs)

    make_inference_fn, params, metrics = algorithm_train(
        **cfg.alg.params,
        environment=env,
        wrap_env_fn=functools.partial(wrap_env_fn, terminal_reward_name=cfg.env.terminal_reward_name) \
            if "terminal_reward_name" in cfg.env else None,
        progress_fn=functools.partial(progress, times, writer),
        seed=cfg.general.seed,
        **kwargs
    )

    if not cfg.general.debug and not cfg.general.no_wandb:
        wandb.finish()

    return make_inference_fn, params, metrics

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    train_with_cfg(cfg)

if __name__=='__main__':
    train()
