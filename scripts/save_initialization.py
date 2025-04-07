import os
import hydra
import importlib

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from jax import config
from brax.envs.fd import get_environment

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    hydra_logdir = HydraConfig.get()["runtime"]["output_dir"]
    param_dir = os.path.join(hydra_logdir, cfg.general.param_dir)
    kwargs = {cfg.alg.checkpoint_keyword_arg: param_dir}

    param_copy_dir = os.path.join(hydra_logdir, "resolved_config.yaml")
    OmegaConf.save(cfg, param_copy_dir, resolve=True)

    alg_module = importlib.import_module(f"brax.training.agents.diffrl_shac.save_initialization")
    algorithm_step = getattr(alg_module, "step")

    param_subset = {}
    for name in [
        "discounting",
        "reward_scaling",
        #"gae_lambda",
        #"unroll_length",
        "batch_size",
        "num_envs",
        "normalize_observations",
        "episode_length",
        "action_repeat",
        "randomization_fn",
        "save_checkpoint_path"
    ]:
        if name in cfg_full["alg"]["params"]:
            param_subset[name] = cfg_full["alg"]["params"][name]

    env = get_environment(cfg.env.name)
    value = algorithm_step(
        **param_subset,
        environment=env,
        **kwargs
    )
    print(value)

if __name__=='__main__':
    train()
