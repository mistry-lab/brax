from brax.envs.ant import Ant

import jax

def main():
    rng = jax.random.PRNGKey(seed=0)
    key_env, key_unroll = jax.random.split(rng)

    env_keys = jax.random.split(key_env, 5)

    ant = Ant()
    ant_state = jax.vmap(ant.reset)(env_keys)

    print(ant_state)
    

if __name__ == '__main__':
    main()