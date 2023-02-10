import math
from typing import Tuple, List, Optional

import numpy as np
import PIL.Image


def np_to_pil(np_array: np.ndarray) -> PIL.Image.Image:
    img = (np_array / np_array.max() * 255).astype(np.int8)[::-1, :]
    return PIL.Image.fromarray(img, mode="L")


class Mosaic:

    def __init__(
            self,
            count: int,
            shape: Tuple[int, int],
            pad: Tuple[int, int] = (2, 2),
            num_x: Optional[int] = None,
            max_width: int = 2000,
    ):
        self.count = count
        self.shape = shape
        self.pad = pad
        self.shape_pad = (self.shape[-2] + self.pad[-2] * 2, self.shape[-1] + self.pad[-1] * 2)
        self.max_width = max_width

        if num_x is not None:
            self.num_x = min(num_x, self.count)
        else:
            edge = int(math.sqrt(self.count))
            self.num_x = int(max(
                min(self.max_width * .75, self.shape_pad[-1] * self.count),
                min(self.max_width, edge * self.shape_pad[-1])
            ) / self.shape_pad[1])

        self.num_y = max(1, int(math.ceil(self.count / self.num_x)))
        self.image = np.zeros((self.shape_pad[-2] * self.num_y, self.shape_pad[-1] * self.num_x))

    def set_image(self, index: int, data: np.ndarray):
        x = index % self.num_x
        y = index // self.num_x
        y_off = y * self.shape_pad[-2] + self.pad[-2]
        x_off = x * self.shape_pad[-1] + self.pad[-1]
        self.image[
            y_off : y_off + data.shape[-1],
            x_off : x_off + data.shape[-2]
        ] = data
