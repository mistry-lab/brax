from brax.envs.ant import Ant
import brax.training.agents.diffrl_shac.networks as shac_networks

import jax
import equinox

def main():
    rng = jax.random.PRNGKey(seed=0)
    key_env, key_unroll = jax.random.split(rng)

    ant = Ant()

    obs_size = ant.observation_size
    action_size = ant.action_size

    networks = shac_networks.make_shac_networks(
        observation_size=obs_size,
        action_size=action_size
    )

    model = networks.policy_network
    params, static = equinox.partition(model, equinox.is_array)
    print(params, static)

if __name__ == '__main__':
    main()
