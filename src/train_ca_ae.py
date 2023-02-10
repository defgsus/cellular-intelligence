import math
import time
from typing import Type, Tuple

import numpy.random
from tqdm import tqdm
import torch
import numpy as np

from src.nn.ae import AutoEncoder
from src.gen import *
from src.util.console import print_2d


device = "cuda"

rng = numpy.random.RandomState()


def random_rule() -> str:
    rule = [set(), set()]
    while not (rule[0] and rule[1]):
        for r in rule:
            prob = rng.random()
            for i in range(9):
                if rng.random() < prob:
                    r.add(i)

    return "01234-016"
    # rule[0] = [1,2,3,4,7,8]
    return "-".join(
        "".join(str(v) for v in r)
        for r in rule
    )


def get_image_batch(
        batch_size: int,
        shape: Tuple[int, int] = (32, 32),
        dtype=torch.float32,
) -> torch.Tensor:
    batch = torch.zeros((batch_size, 1, 32, 32), dtype=dtype)

    for idx in range(batch_size):
        gen = Generator([
            RandomDots(probability=rng.random(), seed=rng),
            CARule(random_rule(), count=rng.randint(10, 70), border="wrap"),
        ])
        batch[idx, 0] = torch.Tensor(gen.generate(shape=shape))

    return batch.to(device)


def train(
        model: torch.nn.Module,
        epochs: int = 1000,
        batch_size: int = 256,
):
    criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=.006, weight_decay=0.0001)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1., weight_decay=0.0001)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.5)

    num_params = sum(
        sum(math.prod(p.shape) for p in g["params"])
        for g in optimizer.param_groups
    )
    print("  trainable params:", num_params)

    last_print_time = time.time()

    for epoch in tqdm(range(epochs)):

        input_batch = get_image_batch(batch_size) * 2 - 1.

        output_batch = model.forward(input_batch)
        # output_batch = torch.round(output_batch)

        loss = criterion(output_batch, input_batch)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        cur_time = time.time()
        if cur_time - last_print_time > 3:
            last_print_time = cur_time

            print_2d(input_batch[0][0].cpu())
            print("-"*32)
            print_2d(output_batch[0][0].cpu())

            print("loss", float(loss))



def main():
    model = AutoEncoder().to(device)

    train(model)


if __name__ == "__main__":
    main()

