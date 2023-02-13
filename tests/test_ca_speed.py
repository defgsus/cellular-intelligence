import unittest
import time
import random
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from src.gen import *
# from src.gen.ca import CARuleFasterQuestionmark


class TestCellularAutomatonSpeed(unittest.TestCase):

    def test_100_speed_per_shape(self):
        for shape in (
                (32, 32),
                (64, 64),
                (256, 256),
                (2048, 2048),
        ):
            count = 20_000_000 // (shape[0] * shape[1])
            gen = Generator([
                RandomDots(),
                CARule(count=count, border="wrap")
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

    def test_200_speed_rules(self):
        rng = random.Random(42)
        rules = list(CARule.iter_automaton_rules())
        rules = [
            rng.choice(rules)
            for i in range(10000)
        ]

        def test(klass):
            shape = (32, 32)
            count = 20
            start_time = time.time()
            pixels = 0
            for rule in tqdm(rules, desc=klass.__name__):
                gen = Generator([
                    klass(count=count, rule=rule, border="wrap")
                ])
                gen.generate(
                    shape=shape,
                    dtype="uint8",
                )
                pixels += count * shape[0] * shape[1]
            run_time = time.time() - start_time

            shape_str = f"{shape[0]}x{shape[1]}"
            pixels_per_sec = int(pixels / run_time)
            print(f"{shape_str:9} x {count * len(rules):7} = {run_time:8.3f}sec {pixels_per_sec:,}p/s")

        test(CARule)
        #test(CARuleFasterQuestionmark)
