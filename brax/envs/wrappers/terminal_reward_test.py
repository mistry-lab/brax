import functools

from absl.testing import absltest
from brax import envs
from brax.envs.wrappers import training
import jax
import jax.numpy as jp
import numpy as np

class TrainingTest(absltest.TestCase):

    def test_domain_randomization_wrapper(self):
        def rand(sys, rng):
            @jax.vmap
            def get_offset(rng):
                offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
                pos = sys.link.transform.pos.at[0].set(offset)
                return pos

            sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
            in_axes = jax.tree.map(lambda x: None, sys)
            in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
            return sys_v, in_axes

        env = envs.create('ant')
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, 256)
        env = training.wrap(
            env,
            episode_length=200,
            randomization_fn=functools.partial(rand, rng=rng),
        )

        # set the same key across the batch for env.reset so that only the
        # randomization wrapper creates variability in the env.step
        key = jp.zeros((256, 2), dtype=jp.uint32)
        state = jax.jit(env.reset)(key)
        self.assertEqual(state.pipeline_state.q[:, 0].shape[0], 256)
        self.assertEqual(np.unique(state.pipeline_state.q[:, 0]).shape[0], 1)

        # test that the DomainRandomizationWrapper creates variability in env.step
        state = jax.jit(env.step)(state, jp.zeros((256, env.sys.act_size())))
        self.assertEqual(state.pipeline_state.q[:, 0].shape[0], 256)
        self.assertEqual(np.unique(state.pipeline_state.q[:, 0]).shape[0], 256)


if __name__ == '__main__':
    absltest.main()
