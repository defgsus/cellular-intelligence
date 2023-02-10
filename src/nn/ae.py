from typing import Callable, Type

import torch

"""
from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""


class Encoder(torch.nn.Module):
    def __init__(
            self,
            num_input_channels: int,
            base_channel_size: int,
            latent_dim: int,
            act_fn: Callable,
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
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            torch.nn.Flatten(),  # Image grid to single feature vector
            torch.nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(torch.nn.Module):
    def __init__(
            self,
            num_input_channels: int,
            base_channel_size: int,
            latent_dim: int,
            act_fn: Callable,
    ):
        super().__init__()
        c_hid = base_channel_size

        self.linear = torch.nn.Sequential(torch.nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            torch.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            torch.nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            torch.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            torch.nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            torch.nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.decoder(x)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(
            self,
            num_input_channels: int = 1,
            base_channel_size: int = 32,
            latent_dim: int = 128,
            act_fn: Callable = torch.nn.ReLU,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_input_channels=num_input_channels,
            base_channel_size=base_channel_size,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )
        self.decoder = Decoder(
            num_input_channels=num_input_channels,
            base_channel_size=base_channel_size,
            latent_dim=latent_dim,
            act_fn=act_fn,
        )

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
