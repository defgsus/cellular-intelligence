import unittest
import time
from typing import Iterable, List

import numpy as np

from src.gen.ca import CARule
from src.gen.base import Generator


class TestCellularAutomatonSpeed(unittest.TestCase):

    def test_speed(self):
        for shape in (
                (64, 64),
                (256, 256),
                (2048, 2048),
        ):
            count = 20_000_000 // (shape[0] * shape[1])
            gen = Generator([

                CARule(count=count)
            ])
            start_time = time.time()
            gen.generate(
                shape=shape,
                dtype="uint8",
            )
            run_time = time.time() - start_time

            shape_str = f"{shape[0]}x{shape[1]}"
            pixels_per_sec = int(count * (shape[0] * shape[1]) / run_time)
            print(f"{shape_str:9} x {count:4} = {run_time:8.3f}sec {pixels_per_sec:,}p/s")
