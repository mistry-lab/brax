from brax.training.average_meter import AverageMeter

import numpy as np
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather

from absl.testing import absltest

def assert_equal(obj, data, expected_data):
    data = process_allgather(data)
    obj.assertTrue(
        jnp.all(data == expected_data), f'Not equal: {data} and {expected_data}'
    )

class AverageMeterTest(absltest.TestCase):

    def test_empty(self) -> None:
        meter = AverageMeter(5)
        assert meter.get_mean() == 0.0

    def test_less_than_max_size(self) -> None:
        meter = AverageMeter(5)
        meter.update(jnp.array([4., 5., 6.]))
        assert_equal(self, meter.get_mean(), jnp.array([5.]))

    def test_equal_max_size(self) -> None:
        meter = AverageMeter(5)
        meter.update(jnp.array([3., 4., 6., 8., 9.]))
        assert_equal(self, meter.get_mean(), jnp.array([6.]))

    def test_greater_than_max_size(self) -> None:
        meter = AverageMeter(5)
        meter.update(jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]))
        assert_equal(self, meter.get_mean(), jnp.array([5.]))

    def test_multiple_insertions(self) -> None:
        meter = AverageMeter(5)

        meter.update(jnp.array([3., 4., 5.]))
        assert_equal(self, meter.get_mean(), jnp.array([4.]))

        meter.update(jnp.array([8., 9., 10.]))
        assert_equal(self, meter.get_mean(), jnp.array([7.]))

    def test_clear(self) -> None:
        meter = AverageMeter(5)

        meter.update(jnp.array([3., 4., 5.]))
        assert_equal(self, meter.get_mean(), jnp.array([4.]))
        meter.clear()

        meter.update(jnp.array([8., 9., 10.]))
        assert_equal(self, meter.get_mean(), jnp.array([9.]))

    def test_batched(self) -> None:
        meter = AverageMeter(5)
        
        meter.update(jnp.array([[3., 8.], [4., 9.], [5., 10.]]))
        assert_equal(self, meter.get_mean(), jnp.array([4., 9.]))

        meter.update(jnp.array([[8., 3.], [9., 4.], [10., 5.]]))
        assert_equal(self, meter.get_mean(), jnp.array([7., 6.]))

if __name__ == '__main__':
    absltest.main()
