from typing import Dict, Tuple, Set, Iterable, Union, Optional

import numpy as np
from sklearn.utils import check_random_state
from .util import total_neighbours
from .base import GeneratorRule


class RandomDots(GeneratorRule):

    def __init__(
            self,
            probability: float = .5,
            seed: Union[None, int, np.random.RandomState] = None,
            shape: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ):
        self.probability = probability
        self.seed = check_random_state(seed)
        self.shape = shape

    def apply(self, state: np.ndarray) -> np.ndarray:
        #if self.shape is None or self.shape is (None, None):
        #    return random_state
        if self.shape is None:
            shape = state.shape
        else:
            shape = [
                min(state.shape[-2], self.shape[-2]) if self.shape[-2] else state.shape[-2],
                min(state.shape[-1], self.shape[-1]) if self.shape[-1] else state.shape[-1],
            ]

        random_state = (
            (self.seed.rand(*shape) < self.probability)
                .astype(state.dtype)
        )

        if random_state.shape == state.shape:
            return random_state

        ox = (state.shape[-1] - shape[-1]) // 2
        oy = (state.shape[-2] - shape[-2]) // 2
        # TODO inplace only
        state[oy:oy + shape[-2], ox:ox + shape[-1]] = random_state

        return state
