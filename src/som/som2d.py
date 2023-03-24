from typing import Tuple, Union, Optional, Sequence
import numpy as np


class SOM2:

    def __init__(
            self,
            n_features: int,
            shape: Sequence[int],
            dtype: Optional[Union[str, np.dtype]] = None,
    ):
        self.map = np.zeros((*shape, n_features), dtype=dtype)
        self._position_map = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_features}, {self.shape}, {self.dtype})"

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns (H, W)
        """
        return self.map.shape[:2]

    @property
    def dtype(self) -> np.dtype:
        return self.map.dtype

    @property
    def n_features(self) -> int:
        return self.map.shape[-1]

    @property
    def position_map(self) -> np.ndarray:
        """
        Return a `H x W x 2` matrix with integer positions for each cell
        """
        if self._position_map is None or self._position_map.shape[:2] != self.shape:
            h, w = self.shape
            # build HxWx2 with y positions
            self._position_map = (
                np.linspace(0, (h,) * w, h, endpoint=False, dtype=self.dtype)
                .reshape((h, w, 1))
                .repeat(2, axis=-1)
            )
            # add x positions
            self._position_map[:, :, 1] = np.linspace(0, (w,) * h, w, endpoint=False, dtype=self.dtype).T

        return self._position_map

    def get_distance_map(self, vector: np.ndarray) -> np.ndarray:
        """
        `H x W` map of distances to single vector of shape (n_features,)
        """
        return np.sqrt(np.sum((self.map - vector) ** 2, axis=-1))

    def get_brush(
            self,
            center: Tuple[int, int],
            radius: float,
            shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Return `H x W x n_features` map with circled brush
        :param center: (y, x)
        :param radius: float
        :param shape: optional
        """
        h, w = shape or self.shape
        y, x = center

        # build `H x W x 2` map of center position
        pos_map = np.tile(np.array((y, x), dtype=self.dtype), h * w).reshape((h, w, 2))
        # euclidean distance to position_map
        dist_map = np.sqrt(np.sum((self.position_map - pos_map) ** 2, axis=-1))
        # invert and apply radius
        brush_map = np.clip(1. - dist_map / radius, 0, 1)
        # expand to feature size
        return brush_map.repeat(self.n_features).reshape((*brush_map.shape, self.n_features))

    def get_best_match(self, vector: np.ndarray) -> Tuple[int, int]:
        """
        Return (y, x) of best matching position on map
        """
        distance_map = self.get_distance_map(vector)

        index = distance_map.argmin()
        return (
            index // self.map.shape[1],
            index % self.map.shape[1]
        )

    def add_vector(
            self,
            vector: np.ndarray,
            center: Tuple[int, int],
            radius: float,
            alpha: float = 1.,
    ):
        """
        Add the vector with shape `n_features` to the map at specified center with radius and alpha.

        :param vector: ndarray of shape `(n_features, )`
        :param center: (y, x)
        :param radius: float
        :param alpha: float [0., 1.]
        """
        brush = self.get_brush(center=center, radius=radius) * alpha

        h, w = self.shape
        vector_map = np.tile(vector, h * w).reshape((h, w, self.n_features))

        # cross-fade to vector
        self.map = (
            (1. - brush) * self.map
            + brush * vector_map
        )
