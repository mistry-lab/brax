import shutil
import wandb
import hydra

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
                        "critic_lr": {
                            "values": [2e-4, 5e-4, 1e-3, 2e-3]
                        },
                        # "tau": {
                        #     "values": [0.2, 0.995]
                        # },
                    }
                },
                "network_factory_params": {
                    "parameters": {
                        "policy_hidden_layer_sizes": {
                            "values": [
                                [64, 64],
                                [256, 128],
                                [512, 256],
                                [128, 64, 32],
                            ],
                        },
                        "value_hidden_layer_sizes": {
                            "values": [
                                [64, 64],
                                [128, 128],
                                [256, 256],
                            ]   
                        }
                    }
                }
            }
        }
    },
}

@hydra.main(config_path="sweep_cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    OmegaConf.update(cfg, "general.sweep", True, force_add=True)
    OmegaConf.update(cfg, "general.no_wandb", False)
    train_with_cfg(cfg)

def main():
    shutil.copytree("cfg", "sweep_cfg", dirs_exist_ok=True)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    wandb.agent(sweep_id, function=train, count=10)

if __name__ == '__main__':
    main()
