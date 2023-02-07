from typing import Dict, Any, Optional, Tuple, Union, Iterable

import numpy as np


class GeneratorRule:

    def apply(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Generator:

    def __init__(
            self,
            rules: Iterable[GeneratorRule],
    ):
        self.rules = list(rules)

    def generate(
            self,
            shape: Optional[Tuple[int, int]] = None,
            dtype: Union[str, np.dtype] = "uint8",
            state: Optional[np.ndarray] = None,
    ):
        assert (shape is None) ^ (state is None), "Must provide `shape` or `state`"

        if state is not None:
            pass
        else:
            state = np.zeros(shape, dtype=dtype)

        return self.apply(state)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply all rules on state
        """
        for rule in self.rules:
            state = rule.apply(state)

        return state
