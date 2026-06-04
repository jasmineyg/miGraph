"""miGraph kernel for multi-instance learning with non-I.I.D. instances."""

import math

import numpy as np

from Prototype import MIL

ZERO_DEGREE_EPSILON = 1e-12


class miGraph(MIL):
    """Compute the miGraph bag kernel and precomputed-kernel CV mappings."""

    def __init__(self, dataset_path, para_gamma: float = 0.1, k: int = 10) -> None:
        super(miGraph, self).__init__(dataset_path)
        self.gamma = para_gamma
        self.k = k
        self.affinity_matrices = self.get_affinity_matrices(self.gamma)
        self.affinity_matrics = self.affinity_matrices

    def get_mapping(self):
        """Yield train/test precomputed-kernel matrices for k-fold evaluation."""
        sim_matrix = np.zeros((self.num_bags, self.num_bags))
        for row in range(self.num_bags):
            for col in range(row, self.num_bags):
                sim = self.sim_between_bags(row, col, self.gamma)
                sim_matrix[row, col] = sim
                sim_matrix[col, row] = sim

        train_idx_dict, test_idx_dict = self.get_index(para_k=self.k)
        for fold_index in range(self.k):
            train_idx = train_idx_dict[fold_index]
            test_idx = test_idx_dict[fold_index]
            train_sim_matrix = sim_matrix[train_idx][:, train_idx]
            test_sim_matrix = sim_matrix[test_idx][:, train_idx]
            yield (
                train_sim_matrix,
                self.bags_label[train_idx],
                test_sim_matrix,
                self.bags_label[test_idx],
                None,
            )

    def get_mapping_by_indices(self, indices, para_k: int = 5):
        """Yield k-fold precomputed-kernel matrices for a subset of bag indexes."""
        num_selected = len(indices)
        sim_matrix = np.zeros((num_selected, num_selected))
        for row in range(num_selected):
            for col in range(row, num_selected):
                sim = self.sim_between_bags(indices[row], indices[col], self.gamma)
                sim_matrix[row, col] = sim
                sim_matrix[col, row] = sim

        temp_rand_idx = np.random.permutation(num_selected)
        folds = np.array_split(temp_rand_idx, para_k)
        indices = np.asarray(indices)

        for fold_index, test_sub in enumerate(folds):
            train_sub = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            train_sim_matrix = sim_matrix[train_sub][:, train_sub]
            test_sim_matrix = sim_matrix[test_sub][:, train_sub]
            tr_lab = self.bags_label[indices[train_sub]]
            te_lab = self.bags_label[indices[test_sub]]
            yield train_sim_matrix, tr_lab, test_sim_matrix, te_lab, None

    def get_affinity_matrices(self, gamma: float):
        """Build one affinity matrix per bag."""
        return [self.bag2matrix(bag_index, gamma) for bag_index in range(self.num_bags)]

    def gaussian_distance(self, u: np.ndarray, v: np.ndarray, gamma: float) -> float:
        """Return ``1 - exp(-gamma * ||u - v||^2)``."""
        diff = u - v
        return 1.0 - math.exp(-gamma * float(np.dot(diff, diff)))

    def bag2matrix(self, i: int, gamma: float) -> np.ndarray:
        """Build the intra-bag binary affinity matrix used by miGraph."""
        instances = self.bags[i, 0][:, : self.dimensions]
        num_instances = len(instances)

        if num_instances == 0:
            return np.zeros((0, 0), dtype=float)
        if num_instances == 1:
            return np.zeros((1, 1), dtype=float)

        diff = instances[:, np.newaxis, :] - instances[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dis_matrix = 1.0 - np.exp(-gamma * dist_sq)
        np.fill_diagonal(dis_matrix, 0.0)

        delta = np.sum(dis_matrix) / (num_instances * (num_instances - 1))
        affinity = (dis_matrix < delta).astype(float)
        np.fill_diagonal(affinity, 0.0)
        return affinity

    def sim_between_bags(self, i: int, j: int, gamma: float) -> float:
        """Return the miGraph similarity between two bags."""
        a_matrix_i = self.affinity_matrices[i]
        b_matrix_j = self.affinity_matrices[j]
        xi = self.bags[i, 0][:, : self.dimensions]
        xj = self.bags[j, 0][:, : self.dimensions]

        if xi.shape[0] == 0 or xj.shape[0] == 0:
            return 0.0

        wi = 1.0 / np.maximum(np.sum(a_matrix_i, axis=1), ZERO_DEGREE_EPSILON)
        wj = 1.0 / np.maximum(np.sum(b_matrix_j, axis=1), ZERO_DEGREE_EPSILON)

        diff = xi[:, np.newaxis, :] - xj[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        kernel = np.exp(-gamma * dist_sq)

        numerator = np.sum((wi[:, np.newaxis] * wj[np.newaxis, :]) * kernel)
        denominator = np.sum(wi) * np.sum(wj)
        return numerator / denominator

    def Gaussian_RBF(self, ins1: np.ndarray, ins2: np.ndarray, gamma: float):
        """Return the Gaussian RBF value between two instances."""
        return np.exp(-gamma * np.sum(np.power((ins1 - ins2), 2)))


__all__ = ["miGraph"]
