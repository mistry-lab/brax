import hydra

import numpy as np
import logging

from train import train_with_cfg
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    OmegaConf.update(cfg, "general.no_wandb", False)

    alg_params = cfg.alg.params
    env_step_per_training_step = (
      alg_params.batch_size * alg_params.unroll_length * alg_params.num_minibatches * alg_params.action_repeat
    )

    num_evals = np.ceil(
        alg_params.num_timesteps / env_step_per_training_step
    ).astype(int)

    logging.info(f"Number of evaluation: {num_evals}")

    cfg.alg.params.num_evals = int(num_evals + 1)

    OmegaConf.update(cfg, "wandb.notes", f"Evaluation run number {cfg.general.seed}", force_add=True)
    train_with_cfg(cfg)

if __name__ == '__main__':
    train()
