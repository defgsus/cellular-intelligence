import math
import os
import shutil
import time
from pathlib import Path
import argparse
from typing import Type, Tuple, Set, Optional, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src.nn.contrastive import *
from src.gen import *
from src.util.console import print_2d

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_PATH / "checkpoints"
TENSORBOARD_PATH = PROJECT_PATH / "runs"


class Trainer:

    def __init__(
            self,
            filename_part: str,
            model: torch.nn.Module,
            datasource: Union[torch.utils.data.DataLoader, object],

            num_epochs: int = 5,
            batch_size: int = 128,

            write_step_interval: int = 1000,
            write_time_interval: float = 30.,

            reset: bool = False,
    ):
        self.filename_part = filename_part
        self.model = model
        self.datasource = datasource
        self.num_samples_per_epoch = len(self.datasource)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.write_step_interval = write_step_interval
        self.write_time_interval = write_time_interval

        self.checkpoint_filename = CHECKPOINT_PATH / f"{filename_part}-snapshot.pt"

        tensorboard_path = Path(TENSORBOARD_PATH / self.filename_part)
        if reset and Path(tensorboard_path).exists():
            shutil.rmtree(tensorboard_path)

        self.writer = SummaryWriter(str(tensorboard_path))
        self.global_step = 0

        if not reset:
            self.load_checkpoint()

        self.num_batches = int(math.ceil(
            (self.num_epochs * self.num_samples_per_epoch - self.global_step) / batch_size
        ))

    def train_step(self):
        raise NotImplementedError

    def write_step(self):
        pass

    def load_checkpoint(self):
        if self.checkpoint_filename.exists():
            checkpoint_data = torch.load(self.checkpoint_filename)
            self.model.load_state_dict(checkpoint_data["state_dict"])
            self.global_step = checkpoint_data["global_step"]

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_filename.parent, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "global_step": self.global_step,
            },
            self.checkpoint_filename,
        )

    def setup_optimizers(self) -> Tuple[list, list]:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=0.0001)
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=1., weight_decay=0.0001)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_batches,
        )
        return [optimizer], [lr_scheduler]

    def log_scalar(self, tag: str, value):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.global_step)

    def log_image(self, tag: str, image: torch.Tensor):
        self.writer.add_image(tag=tag, img_tensor=image, global_step=self.global_step)

    def train(self):

        optimizers, schedulers = self.setup_optimizers()

        num_params = sum(
            sum(math.prod(p.shape) for p in g["params"])
            for g in optimizers[0].param_groups
        )
        print("  trainable params:", num_params)

        last_write_time = time.time()
        last_write_step = self.global_step
        last_save_loss = None
        for batch_idx in tqdm(range(self.num_batches)):

            loss = self.train_step()

            self.model.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

            self.global_step += self.batch_size

            loss = float(loss)
            self.log_scalar("train_loss", loss)
            for sched in schedulers:
                self.log_scalar(f"learnrate.{sched.__class__.__name__}", sched.get_last_lr()[0])

            cur_time = time.time()
            if (
                    batch_idx == 0
                    or self.global_step - last_write_step >= self.write_step_interval
                    or cur_time - last_write_time >= self.write_time_interval
            ):
                last_write_step += self.write_step_interval
                last_write_time = cur_time

                if batch_idx != 0:
                    if last_save_loss is None or loss <= last_save_loss:
                        last_save_loss = loss
                        self.save_checkpoint()

                    self.write_step()

        self.writer.close()

