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
        if count is None:
            count = self.count

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
                # Note:
                # also tried the following which is not faster, rather a bit slower:
                #   match = dead & (neigh == num_n)
                #   new_state[match] = 1

            data = new_state.astype(data.dtype)

        return data

    @classmethod
    def iter_automaton_rules(cls):
        for i in range(2 ** 18):
            r1 = i & (2 ** 9-1)
            r2 = (i >> 9) & (2**9-1)
            r1 = [b for b in range(9) if (r1 >> b) & 1]
            r2 = [b for b in range(9) if (r2 >> b) & 1]
            if r2:
                yield "-".join((
                    "".join(str(r) for r in r1),
                    "".join(str(r) for r in r2),
                ))


class CA2Rule(GeneratorRule):

    def __init__(
            self,
            rule: Tuple[Dict[int, int], Dict[int, int]] = ({3:1}, {2:1, 3:1}),
            count: int = 1,
            border: str = "zero",
    ):
        assert border in ("zero", "wrap", "symm")

        self.rule = rule
        self.count = count
        self.border = border

    def apply(self, data: np.ndarray, count: Optional[int] = None, border: Optional[str] = None) -> np.ndarray:
        if border is None:
            border = self.border
        if count is None:
            count = self.count

        for i in range(count):
            neigh = total_neighbours(data, border)
            dead = data == 0
            alive = np.invert(dead)

            new_state = np.zeros(data.shape, dtype=data.dtype)

            if self.rule[1]:
                for num_n, value in self.rule[0].items():
                    match = dead & (neigh == num_n)
                    new_state[match] = value
                for num_n, value in self.rule[1].items():
                    match = alive & (neigh == num_n)
                    new_state[match] = value

            data = new_state.astype(data.dtype)

        return data
