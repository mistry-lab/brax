import shutil
import wandb
import hydra
import yaml

from train import train_with_cfg
from omegaconf import DictConfig, OmegaConf

sweep_configuration = {
    "name": "sweepdemo",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "eval/episode_reward"},
    "parameters": {
        "alg": {
            "parameters": {
                "params": {
                    "parameters": {
                        "actor_xi": {
                            "values": [0.05 * i for i in range(21)]
                        },
                    }
                },
            }
        }
    },
}

@hydra.main(config_path="sweep_cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    OmegaConf.update(cfg, "general.sweep", True, force_add=True)
    OmegaConf.update(cfg, "general.no_wandb", False)

    OmegaConf.update(cfg, "alg.params.batch_size", cfg.alg.params.num_envs // cfg.alg.params.num_minibatches)
    train_with_cfg(cfg)

def main():
    shutil.copytree("cfg", "sweep_cfg", dirs_exist_ok=True)
    with open("sweep_cfg/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=cfg["wandb"]["project"])
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()
