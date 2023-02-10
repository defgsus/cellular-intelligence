from typing import Callable, Type

import torch


class ContrastiveEncoder(torch.nn.Module):
    def __init__(
            self,
            num_input_channels: int = 1,
            base_channel_size: int = 32,
            latent_dim: int = 64,
            act_fn: Callable = torch.nn.GELU,
    ):
        super().__init__()
        c_hid = base_channel_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            torch.nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            torch.nn.Conv2d(2 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            torch.nn.Conv2d(3 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * c_hid * 4*4, latent_dim),
            #torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)

    def XX_init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            #m.weight.data.fill_(1.0)
            m.weight *= .02


class ContrastiveEncoder2(torch.nn.Module):
    def __init__(
            self,
            num_input_channels: int = 1,
            base_channel_size: int = 32,
            latent_dim: int = 128,
            act_fn: Callable = torch.nn.GELU,
    ):
        super().__init__()
        c_hid = base_channel_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channels, c_hid, kernel_size=5, padding=1, stride=1),  # 32x32 => 16x16
            act_fn(),
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=7, padding=1),
            act_fn(),
            torch.nn.Conv2d(c_hid, 2 * c_hid, kernel_size=9, padding=1, stride=1),  # 16x16 => 8x8
            act_fn(),
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=11, padding=1),
            act_fn(),
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=7, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 4*4 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def XX_init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            #m.weight.data.fill_(1.0)
            m.weight *= .4
