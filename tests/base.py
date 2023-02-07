import unittest
from typing import Any

import numpy as np


class TestBase(unittest.TestCase):

    def assertEqual(self, first: Any, second: Any, msg: Any = ...) -> None:
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(second, np.ndarray):
            second = second.tolist()

        super().assertEqual(first, second, msg)