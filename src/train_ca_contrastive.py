import math
import os
import shutil
import time
from pathlib import Path
import argparse
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Type, Tuple, Set, Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src.nn.trainer import Trainer, PROJECT_PATH, CHECKPOINT_PATH
from src.nn.contrastive import *
from src.gen import *
from src.util.console import print_2d


device = "cuda" if torch.cuda.is_available() else "cpu"

rng = np.random.RandomState()

RuleType = Tuple[Set[int], Set[int]]


def random_rule() -> RuleType:
    rule = (set(), set())
    while not (rule[0] and rule[1]):
        for r in rule:
            prob = rng.random()
            for i in range(9):
                if rng.random() < prob:
                    r.add(i)
    return rule


def rule_str(rule: RuleType) -> str:
    return "-".join(
        "".join(str(v) for v in r)
        for r in rule
    )


def get_anti_rule(rule: RuleType) -> RuleType:
    new_rule = (set(), set())
    for r, nr in zip(rule, new_rule):
        for i in range(9):
            if i not in r or rng.random() < .1:
                nr.add(i)

    return new_rule


class DataSource:

    def __init__(self):
        self.rules = np.array(list(CARule.iter_automaton_rules()))
        self.rng = np.random.RandomState(23)
        # self.rules = np.random.choice(self.rules, (5, ))

    def __len__(self):
        return len(self.rules) #* (3000 // 5)

    def get_random_batch(
            self,
            batch_size: int,
            shape: Tuple[int, int] = (32, 32),
            dtype=torch.float32,
            num_repeat: int = 3,
    ):
        images = torch.zeros((batch_size, 1, shape[0], shape[1]), dtype=dtype)
        augmented = [
            torch.zeros((batch_size, 1, shape[0], shape[1]), dtype=dtype)
            for i in range(num_repeat)
        ]
        negative = torch.zeros((batch_size, 1, shape[0], shape[1]), dtype=dtype)

        for idx in range(batch_size):
            rule = self.rng.choice(self.rules)

            gen = Generator([
                RandomDots(probability=rng.random(), seed=rng),
                CARule(rule, count=rng.randint(20, 70), border="wrap"),
            ])
            images[idx, 0] = torch.Tensor(gen.generate(shape=shape))
            for i in range(num_repeat):
                augmented[i][idx, 0] = torch.Tensor(gen.generate(shape=shape))

            neg_rule = rule
            while neg_rule == rule:
                neg_rule = self.rng.choice(self.rules)
            gen = Generator([
                RandomDots(probability=rng.random(), seed=rng),
                CARule(neg_rule, count=rng.randint(20, 70), border="wrap"),
            ])
            negative[idx, 0] = torch.Tensor(gen.generate(shape=shape))

        return (
            images,
            augmented,
            negative,
        )


class ProcessDatasource():
    def __init__(self, database, num_processes: int = 4):
        self.database = database
        self.processes = [
            Process(name=f"dataprocess-{i}", target=self._process_loop)
            for i in range(num_processes)
        ]
        self.manager = Manager()
        self.action_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.batches = []
        self.num_batches_requested = 0

        for p in self.processes:
            p.start()

    def stop(self):
        self.action_queue.put_nowait({"STOP": True})
        for p in self.processes:
            p.join()

    def __len__(self):
        return len(self.database)

    def get_random_batch(self, **kwargs):
        while True:
            while True:
                try:
                    self.batches.append(self.result_queue.get_nowait())
                    self.num_batches_requested -= 1
                except Empty:
                    break

            for i, [batch_kwargs, batch] in enumerate(self.batches):
                if kwargs == batch_kwargs:
                    self.batches.pop(i)
                    return batch

            while self.num_batches_requested < len(self.processes):
                self.action_queue.put_nowait(kwargs)
                self.num_batches_requested += 1

            time.sleep(.1)

    def _process_loop(self):
        while True:
            try:
                kwargs = self.action_queue.get(timeout=1)
                if kwargs.get("STOP"):
                    self.action_queue.put_nowait({"STOP": True})
                    return

                batch = self.database.get_random_batch(**kwargs)
                self.result_queue.put_nowait((kwargs, batch))

                self.action_queue.task_done()
            except Empty:
                pass


class ContrastiveTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_labels = torch.ones(self.batch_size).to(device)
        self.negative_labels = -self.positive_labels

    def setup_optimizers(self) -> Tuple[list, list]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.002, weight_decay=0.0001)
        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=0.0001)
        #optimizer = torch.optim.Adadelta(self.model.parameters(), lr=1., weight_decay=0.0001)
        #optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0002, momentum=0.5)
        #optimizer = torch.optim.ASGD(self.model.parameters(), lr=.5, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_batches,
        )
        return [optimizer], [lr_scheduler]

    def train_step(self):
        batch = self.datasource.get_random_batch(batch_size=self.batch_size)
        batch = [
            t.to(device) * 2. - 1. if isinstance(t, torch.Tensor) else [
                t2.to(device) * 2. - 1.
                for t2 in t
            ]
            for t in batch
        ]
        self.images, self.augmented_images, self.negative_images = batch

        self.image_embedding = self.model.forward(self.images)
        self.aug_image_embedding = [
            self.model.forward(aug_images)
            for aug_images in self.augmented_images
        ]
        self.neg_image_embedding = self.model.forward(self.negative_images)

        positive_loss = 0.
        for aug_emb in self.aug_image_embedding:
            positive_loss += F.cosine_embedding_loss(
                self.image_embedding, aug_emb,
                target=self.positive_labels,
            )
        # positive_loss /= len(self.augmented_images)

        negative_loss = F.cosine_embedding_loss(
            self.image_embedding, self.neg_image_embedding,
            target=self.negative_labels,
        )

        #embedding_loss = .01 * F.mse_loss(image_embedding.abs().mean(), desired_embedding_mean)

        self.log_scalar("train_loss_positive", positive_loss)
        self.log_scalar("train_loss_negative", negative_loss)
        self.log_scalar("embedding_abs_mean", self.image_embedding.abs().mean())

        loss = negative_loss + positive_loss
        return loss

    def write_step(self):
        self.log_image(
            "input",
            make_grid(torch.cat([
                self.images[:8],
                ] + [
                    aug_images[:8]
                    for aug_images in self.augmented_images
                ] + [
                self.negative_images[:8],
            ]))
        )
        self.log_image(
            "embeddings",
            make_grid(
                [
                    self.image_embedding.reshape(1, *self.image_embedding.shape),
                ] + [
                    aug_emb.reshape(1, *aug_emb.shape)
                    for aug_emb in self.aug_image_embedding
                ] + [
                    self.neg_image_embedding.reshape(1, *self.neg_image_embedding.shape),
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", type=str,
        help="name of the output files for checkpoints and tensorbord logs"
    )
    parser.add_argument(
        "-r", "--reset", type=bool, nargs="?", default=False, const="True",
        help="Delete previous checkpoint"
    )
    args = parser.parse_args()

    name = args.name
    model = ContrastiveEncoder(
        base_channel_size=128,  # was 32
    ).to(device)
    if hasattr(model, "init_weights"):
        with torch.no_grad():
            model.apply(model.init_weights)

    trainer = ContrastiveTrainer(
        model=model,
        filename_part=name,
        datasource=ProcessDatasource(DataSource(), num_processes=8),
        reset=args.reset,
        num_epochs=10,
    )
    trainer.train()
    trainer.writer.close()


def test_batch_speed():
    batcher = DataSource()  # bs=128 -> ~2sec per batch -> 100b == 3.20min
    batcher = ProcessDatasource(batcher, num_processes=8)  # -> 100b == ~40sec

    try:
        for i in tqdm(range(100)):
            batch = batcher.get_random_batch(batch_size=128)
            assert batch
    finally:
        batcher.stop()


if __name__ == "__main__":
    main()
    #test_batch_speed()
