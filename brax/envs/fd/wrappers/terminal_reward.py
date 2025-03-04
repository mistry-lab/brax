from brax import base
from flax import struct

import jax

@struct.dataclass
class TerminalReward(base.Base):
    reward: jax.Array
    metrics: jax.Array
