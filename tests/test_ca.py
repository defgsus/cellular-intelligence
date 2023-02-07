import unittest
from typing import Iterable, List

import numpy as np

from src.gen.ca import CARule


class TestCellularAutomaton(unittest.TestCase):

    def assert_ca(
            self,
            input_and_output: str,
            rule: str = "3-23",
            border: str = "zero",
    ):
        def _convert(lines: List[str], index: int) -> np.ndarray:
            return np.array([
                [1 if c == "#" else 0 for c in line.split("|")[index]]
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

        ca = CARule(rule, border=border)
        state = input
        for i, expected_output in enumerate(expected_outputs):
            state = ca.apply(state)
            self.assertEqual(
                expected_output.tolist(),
                state.tolist(),
                f"\nExpected (in iteration {i}):\n{expected_output}\nGot:\n{state}"
            )

    def test_game_of_life(self):
        self.assert_ca("""
            .....|.....|..#..
            ..#..|.###.|.#.#.
            .###.|.#.#.|#...#
            ..#..|.###.|.#.#.
            .....|.....|..#..
        """)
        self.assert_ca("""
            .#...|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....|.....
            ..#..|#.#..|..#..|.#...|..#..|.....|.....|.....|.....|.....|.....|.....|.....
            ###..|.##..|#.#..|..##.|...#.|.#.#.|...#.|..#..|...#.|.....|.....|.....|.....
            .....|.#...|.##..|.##..|.###.|..##.|.#.#.|...##|....#|..#.#|....#|...##|...##
            .....|.....|.....|.....|.....|..#..|..##.|..##.|..###|...##|...##|...##|...##
        """)
        self.assert_ca("""
            .#...|.....|.....|.....|.....|.....|.....|.....|.....|...#.|...##|...##|#..##|#...#|#..#.|##...|.#...
            ..#..|#.#..|..#..|.#...|..#..|.....|.....|.....|.....|.....|.....|.....|.....|....#|#...#|#...#|##..#
            ###..|.##..|#.#..|..##.|...#.|.#.#.|...#.|..#..|...#.|.....|.....|.....|.....|.....|.....|.....|.....
            .....|.#...|.##..|.##..|.###.|..##.|.#.#.|...##|....#|..#.#|....#|...#.|....#|.....|.....|.....|.....
            .....|.....|.....|.....|.....|..#..|..##.|..##.|..###|...##|..#.#|#...#|#....|#..#.|#....|....#|#....
        """, border="wrap")
