import argparse
from typing import Tuple

import numpy as np

from src.gen import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rule", type=str,
        help="CA rule",
    )
    parser.add_argument(
        "-c", "--count", type=int, nargs="?", default=10,
    )
    parser.add_argument(
        "-rp", "--random-prob", type=float, nargs="?", default=.1,
    )
    parser.add_argument(
        "-s", "--size", type=str, nargs="?", default="32x128",
    )
    params = vars(parser.parse_args())
    params["size"] = tuple(int(s) for s in params["size"].split("x"))
    return params


def dump_state(state: np.ndarray, file=None):
    for row in state:
        print("".join("#" if v else "." for v in row), file=file)


def main(
        rule: str,
        count: int,
        random_prob: float,
        size: Tuple[int, int],
        border: str = "wrap",
):
    gen = Generator([
        RandomDots(probability=random_prob),
        CARule(rule=rule, count=count, border=border),
        #CARule(rule="08-17", iterations=count, border=border),
    ])
    gen2 = Generator([
        CARule(rule=rule, count=1, border=border)
    ])

    state = gen.generate(shape=size)
    dump_state(state)

    try:
        step_size = 1
        while True:
            inp = input().lower()
            if inp == "q":
                break
            try:
                step_size = int(inp)
            except ValueError:
                pass
            for i in range(step_size):
                state = gen2.apply(state)
            dump_state(state)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(**parse_args())

