import time
import yaml

import jax
import mujoco
from mujoco import viewer

from brax.envs.fd import get_environment

def main():
    with open("visualize_paths", "r") as f:
        paths = [line.strip() for line in f.readlines()]
        config_path, policy_parameters_path = paths

    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)
        env = get_environment(run_config["env"]["name"])

        rng = jax.random.PRNGKey(0)
        mjx_data = env.reset(rng).pipeline_state
        data = mujoco.MjData(env.sys.mj_model)

        data.qpos[:] = mjx_data.qpos
        data.qvel[:] = mjx_data.qvel
        mujoco.mj_forward(env.sys.mj_model, data)

        with viewer.launch_passive(env.sys.mj_model, data) as v:
            
            for i in range(1000):
                mujoco.mj_step(env.sys.mj_model, data)
                v.sync()
                time.sleep(0.01)

if __name__ == '__main__':
    main()
