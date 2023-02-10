import unittest
from typing import Iterable, List

import numpy as np

from src.gen.ca import CA2Rule


class TestCellularAutomaton2(unittest.TestCase):

    def assert_ca(
            self,
            input_and_output: str,
            rule = ({3: 1}, {2: 1, 3: 1}),
            border: str = "zero",
    ):
        def _convert(lines: List[str], index: int) -> np.ndarray:
            return np.array([
                [int(c) if c.isnumeric() else 0 for c in line.split("|")[index]]
                for line in lines
            ], dtype="uint8")

        lines = [
            line.strip() for line in input_and_output.splitlines()
            if line.strip()
        ]

        input = _convert(lines, 0)
        expected_outputs = [
            _convert(lines, i + 1)
            for i in range(0, lines[0].count("|"))
        ]

        ca = CA2Rule(rule, border=border)
        state = input
        for i, expected_output in enumerate(expected_outputs):
            state = ca.apply(state)
            self.assertEqual(
                expected_output.tolist(),
                state.tolist(),
                f"\nExpected (in frame {i+2}):\n{expected_output}\nGot:\n{state}"
            )

    def test_game_of_life(self):
        self.assert_ca("""
            .....|.....|..1..
            ..1..|.111.|.1.1.
            .111.|.1.1.|1...1
            ..1..|.111.|.1.1.
            .....|.....|..1..
        """)
        self.assert_ca("""
            .#...|.....|.....|.....|.....|.....|.....|.....|.....|...#.|...##|...##|#..##|#...#|#..#.|##...|.#...
            ..#..|#.#..|..#..|.#...|..#..|.....|.....|.....|.....|.....|.....|.....|.....|....#|#...#|#...#|##..#
            ###..|.##..|#.#..|..##.|...#.|.#.#.|...#.|..#..|...#.|.....|.....|.....|.....|.....|.....|.....|.....
            .....|.#...|.##..|.##..|.###.|..##.|.#.#.|...##|....#|..#.#|....#|...#.|....#|.....|.....|.....|.....
            .....|.....|.....|.....|.....|..#..|..##.|..##.|..###|...##|..#.#|#...#|#....|#..#.|#....|....#|#....
        """, border="wrap")

    def test_something_else(self):
        self.assert_ca("""
            .....|.....|.....|22.22|11.11|.....
            .....|.222.|.1.1.|23.32|1...1|.....
            ..1..|.232.|.....|.....|.....|.....
            .....|.222.|.1.1.|23.32|1...1|.....
            .....|.....|.....|22.22|11.11|.....
        """, rule=({1:2}, {0:3, 7:1}))

        self.assert_ca("""
            .....|.....|.2.2.|.....
            ..1..|.212.|2.3.2|.....
            .111.|.1.1.|.3.3.|.....
            ..1..|.212.|2.3.2|.....
            .....|.....|.2.2.|.....
        """, rule=({3: 2}, {3: 1, 6: 3}))
