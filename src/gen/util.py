from typing import Dict, Tuple, Set, Iterable, Union, Optional

import numpy as np
import scipy.signal


_total_neighbour_kernels: Dict[str, np.ndarray] = {}


def total_neighbours(
        data: np.ndarray,
        border: str = "zero",
) -> np.ndarray:
    assert border in ("zero", "wrap", "symm")

    if border == "zero":
        border = "fill"

    if data.dtype not in _total_neighbour_kernels:
        _total_neighbour_kernels[data.dtype] = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=data.dtype)

    return scipy.signal.convolve2d(
        data,
        _total_neighbour_kernels[data.dtype],
        mode="same",
        boundary=border,
    )
