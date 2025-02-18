from typing import NamedTuple

from brax.training.acme.types import NestedArray

class ValueTrainingSample(NamedTuple):
    observation: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    truncation: NestedArray
