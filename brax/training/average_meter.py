import jax.numpy as jnp
import jax

class AverageMeter:
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.shape[0]
        if size == 0:
            return
        new_mean = jnp.mean(values, axis=0)
        size = jnp.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
