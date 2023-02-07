from typing import Optional, Union, Callable

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise


class AutoCluster(ClusterMixin, BaseEstimator):

    SUPPORTED_DISTANCE_METRICS = [
        "euclidean", "l2", "l1", "manhattan", "cityblock",
        "braycurtis", "canberra", "chebyshev", "correlation",
        "cosine", "dice", "hamming", "jaccard", "kulsinski",
        "matching", "minkowski", "rogerstanimoto",
        "russellrao", "sokalmichener",
        "sokalsneath", "sqeuclidean", "yule",
        "nan_euclidean",
    ]

    def __init__(
            self,
            distance_threshold: float = 1.,
            distance_metric: Union[str, Callable] = "euclidean",
            max_n_clusters: int = 10,
            label_dtype: Union[str, np.dtype] = "int64",
            # random_state=None,
            verbose: bool = False,
    ):
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric
        self.max_n_clusters = max_n_clusters
        self.label_dtype = label_dtype
        # self.random_state = random_state
        self.verbose = verbose
        self.cluster_centers_: Optional[np.ndarray] = None
        self.cluster_counts_: Optional[np.ndarray] = None
    
    @property
    def n_clusters_(self) -> int:
        if self.cluster_centers_ is None:
            return 0
        return self.cluster_centers_.shape[0]

    def fit(self, X: np.ndarray, y=None) -> "AutoCluster":
        self.cluster_centers_ = None
        self.cluster_counts_ = None

        self.partial_fit(X, y)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.partial_fit(X, y, return_labels=True)

    def partial_fit(
            self,
            X: np.ndarray,
            y=None,
            return_labels: bool = False,
    ) -> Union["AutoCluster", np.ndarray]:
        X = self._validate_data(
            X, y,
            dtype=[np.float64, np.float32, np.bool_],
            reset=self.cluster_centers_ is None,
        )
        # random_state = check_random_state(self.random_state)
        if return_labels:
            labels = np.ndarray((X.shape[0], ), dtype=self.label_dtype)

        label_idx = 0
        while X.shape[0]:
            if self.cluster_centers_ is None:
                # create first cluster
                self.cluster_centers_ = np.copy(X[0]).reshape(1, -1)
                self.cluster_counts_ = np.array([1], dtype=self.label_dtype)
                X = X[1:]
                if return_labels:
                    labels[label_idx] = 0
                    label_idx += 1
                continue

            distances = self._distance_metric(X)
            best_ids = np.argmin(distances, axis=-1)
            best_distances = np.min(distances, axis=-1)

            for idx, (feature, cluster_id, distance) in enumerate(zip(X, best_ids, best_distances)):
                if distance > self.distance_threshold and self.n_clusters_ < self.max_n_clusters:
                    # create a new cluster
                    cluster_id = self.n_clusters_
                    self.cluster_centers_ = np.resize(
                        self.cluster_centers_, (cluster_id + 1, self.cluster_centers_.shape[1])
                    )
                    self.cluster_counts_ = np.resize(self.cluster_counts_, (cluster_id + 1, ))
                    self.cluster_centers_[cluster_id] = feature
                    self.cluster_counts_[cluster_id] = 1
                    X = X[idx + 1:]
                    if return_labels:
                        labels[label_idx] = cluster_id
                        label_idx += 1
                    # start again with new distance matrix
                    break

                self._add_to_cluster(cluster_id, feature)
                if return_labels:
                    labels[label_idx] = cluster_id
                    label_idx += 1

            if idx == X.shape[0] - 1:
                break

        if return_labels:
            return labels
        else:
            return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(
            X,
            dtype=[np.float64, np.float32, np.bool_],
            reset=False,
        )
        distances = self._distance_metric(X)
        best_ids = np.argmin(distances, axis=-1)
        return best_ids

    def _add_to_cluster(self, cluster_id: int, feature: np.ndarray):
        count = self.cluster_counts_[cluster_id]
        if self.cluster_centers_.dtype == np.bool_:
            self.cluster_centers_[cluster_id] = (
                ((
                    self.cluster_centers_[cluster_id].astype("float64") * count
                    + feature.astype("float64")
                ) / (count + 1)).astype("bool")
            )
        else:
            self.cluster_centers_[cluster_id] = (
                (self.cluster_centers_[cluster_id] * count + feature) / (count + 1)
            )
        self.cluster_counts_[cluster_id] = count + 1

    def _distance_metric(self, X: np.ndarray):
        return pairwise.pairwise_distances(
            X,
            self.cluster_centers_,
            metric=self.distance_metric,
        )

