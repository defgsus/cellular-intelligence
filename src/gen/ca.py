from typing import Dict, Tuple, Set, Iterable, Union, Optional

import numpy as np

from .util import total_neighbours
from .base import GeneratorRule


class CARule(GeneratorRule):

    def __init__(
            self,
            rule: Union[
                str,
                Tuple[Iterable[int], Iterable[int]],
            ] = "3-23",
            count: int = 1,
            border: str = "zero",
    ):
        assert border in ("zero", "wrap", "symm")

        if isinstance(rule, str):
            r1, r2 = rule.split("-")
        else:
            r1, r2 = rule
        self.rule: Tuple[Set[int], Set[int]] = (
            {int(r) for r in r1},
            {int(r) for r in r2},
        )

        self.count = count
        self.border = border

    def apply(self, data: np.ndarray, count: Optional[int] = None, border: Optional[str] = None) -> np.ndarray:
        if border is None:
            border = self.border
        count = count if count is not None else self.count

        for i in range(count):
            neigh = total_neighbours(data, border)
            dead = data == 0
            alive = np.invert(dead)

            new_state = np.zeros(data.shape, dtype="bool")

            if self.rule[1]:
                for num_n in self.rule[0]:
                    new_state |= dead & (neigh == num_n)
                for num_n in self.rule[1]:
                    new_state |= alive & (neigh == num_n)

            data = new_state.astype(data.dtype)

        return data
