import time
import os
import subprocess
import pathlib
import importlib
import yaml
import json
import argparse
import logging
import sys

import argcomplete
from dotenv import load_dotenv
from paramiko import SSHClient
import mujoco
from mujoco import viewer

import jax
from brax.envs.fd import get_environment

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

remote_config_filename = "resolved_config.yaml"
script_dir = os.path.dirname(os.path.realpath(__file__))

config_dir = os.path.join(script_dir, "remote_cfg")
param_dir = os.path.join(script_dir, "params")

def get_most_recent_parameters_path(address: str, username: str, password: str):
    client = SSHClient()
    client.load_system_host_keys()
    client.connect(address, username=username, password=password)
    stdin, stdout, stderr = client.exec_command('./brax/brax/scripts/newest_train_parameters.sh')
    return pathlib.Path(stdout.read().strip().decode("utf-8"))

def get_config(address: str, remote_training_dir: str, output_filename: str, password: str | None):
    output_config = os.path.join(config_dir, output_filename)
    os.makedirs(output_config, exist_ok=True)
    subprocess.call(["./sync_config.sh", remote_training_dir / remote_config_filename, output_config, address, password])

    return os.path.join(output_config, remote_config_filename)

def get_most_recent_parameters(
        address: str,
        remote_checkpoint_dir: str,
        output_filename: str,
        network_config_filename: str,
        username: str, 
        password: str | None):
    local_params = os.path.join(param_dir, output_filename)
    os.makedirs(local_params, exist_ok=True)

    alg_config_file = remote_checkpoint_dir.parent / network_config_filename
    scp_command = f"scp -r {username}@{address}:{remote_checkpoint_dir} {local_params}"
    alg_scp_command =  f"scp {username}@{address}:{alg_config_file} {local_params}"
    if password:
        scp_command = f"sshpass -p {password} {scp_command}"
        alg_scp_command = f"sshpass -p {password} {alg_scp_command}"

    subprocess.call([
        f"script -q -c '{scp_command}' </dev/null",
    ], shell=True)

    subprocess.call([
        f"script -q -c '{alg_scp_command}' </dev/null",
    ], shell=True)

    policy_params_name = remote_checkpoint_dir.name

    return \
        os.path.join(local_params, policy_params_name)

def visualise_traj_generic(
    x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01
):
    with viewer.launch_passive(m, d) as v:
        for dx in x:
            step_start = time.time()
            d.qpos[:] = dx.qpos
            d.qvel[:] = dx.qvel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def generate_trajectory(env_reset, env_step, inference_fn, seed):
    rollout = []

    seed = 0 if seed is None else int(seed)
    rng = jax.random.PRNGKey(seed=seed)
    state = env_reset(rng=rng)
    for _ in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = inference_fn(state.obs, act_rng)
        state = env_step(state, act)

    return rollout

def get_command_line_args():
    parser = argparse.ArgumentParser(description='FD visualization script.')
    argcomplete.autocomplete(parser)
    parser.add_argument('-a', '--address', help='Address from which to scp config and parameters.')
    parser.add_argument('--password', help='Password used for SSH connection.')

    parser.add_argument('-p','--param-path', help='Path to policy parameters.')
    parser.add_argument('-c','--config-path', help='Path to training configuration.')
    
    parser.add_argument('-s', '--seed', help='Seed used to initialize the environment.')

    args = parser.parse_args()

    return args

def main():
    load_dotenv()
    args = get_command_line_args()

    run_config = None

    if args.seed:
        with open("visualize_paths", "r") as f:
            paths = [line.strip() for line in f.readlines()]
            config_path, policy_parameters_path = paths
    else:
        address = args.address if args.address else os.environ["ADDRESS"]  
        username = "alexaldermanwebb"
        path = get_most_recent_parameters_path(address, username, args.password)

        train_dir = path.parents[1]
        isotime = f"{train_dir.parents[0].name}T{train_dir.name}"
        logging.info(f"Loading training run in {isotime}")
        
        config_path = get_config(address, train_dir, isotime, args.password)

        with open(config_path, "r") as f:
            run_config = yaml.safe_load(f)  

        policy_parameters_path = get_most_recent_parameters(
            address,
            path,
            isotime,
            run_config["alg"]["network_config_filename"],
            username, 
            args.password
        )

        sharding_file = os.path.join(policy_parameters_path, "_sharding")
        with open(sharding_file, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                parsed_value = json.loads(value)
                parsed_value["device_str"] = str(jax.local_devices()[0])
                data[key] = json.dumps(parsed_value)

        pathlib.Path(sharding_file).write_text(json.dumps(data))

    if not run_config:
        with open(config_path, "r") as f:
            run_config = yaml.safe_load(f) 

    checkpoint = importlib.import_module(run_config["alg"]["checkpoint_path"])

    env = get_environment(run_config["env"]["name"])
    jit_env_reset = jax.jit(env.reset)

    load_policy = getattr(checkpoint, "load_policy")
    inference_fn = load_policy(path=policy_parameters_path)

    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    trajectory = generate_trajectory(jit_env_reset, jit_env_step, jit_inference_fn, args.seed)

    with open("visualize_paths", "w") as f:
        f.writelines(
            [f"{config_path}\n", policy_parameters_path]
        )

    data = mujoco.MjData(env.sys.mj_model)
    visualise_traj_generic(trajectory, data, env.sys.mj_model)

if __name__=='__main__':
    main()
