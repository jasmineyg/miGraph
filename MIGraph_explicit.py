"""Explicit MIGraph kernel with node and edge features."""

import math

import numpy as np

from Prototype import MIL

EDGE_WEIGHT_EPSILON = 1e-8
KERNEL_NORMALIZATION_EPSILON = 1e-8
EDGE_KERNEL_CHUNK_SIZE = 2000


class MIGraph(MIL):
    """Compute explicit MIGraph kernels and precomputed-kernel CV mappings."""

    def __init__(
        self,
        dataset_path,
        para_gamma_node: float = 0.1,
        para_gamma_edge: float = 0.1,
        para_epsilon=None,
        k: int = 10,
    ) -> None:
        super(MIGraph, self).__init__(dataset_path)
        self.gamma_node = para_gamma_node
        self.gamma_edge = para_gamma_edge
        self.epsilon = para_epsilon
        self.k = k
        self.graphs = [self.build_graph(i) for i in range(self.num_bags)]
        self.kG_self = [self._kG_unnormalized(i, i) for i in range(self.num_bags)]
        self.kernel_matrix = self.compute_kernel_matrix()

    def build_graph(self, i: int):
        """Build one bag graph with node features, edges, and edge features."""
        nodes = self.bags[i, 0][:, : self.dimensions]
        num_nodes = nodes.shape[0]

        if num_nodes == 0:
            return {"nodes": nodes, "edges": [], "edge_features": []}

        diff = nodes[:, None, :] - nodes[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))

        if self.epsilon is not None:
            epsilon = self.epsilon
        elif num_nodes > 1:
            triu_idx = np.triu_indices(num_nodes, k=1)
            pair_distances = distances[triu_idx]
            epsilon = np.mean(pair_distances) if pair_distances.size > 0 else 0.0
        else:
            epsilon = 0.0

        edge_list = []
        raw_weights = []
        for source in range(num_nodes):
            for target in range(source + 1, num_nodes):
                if 0.0 < distances[source, target] < epsilon:
                    edge_list.append((source, target))
                    raw_weights.append(
                        1.0 / (distances[source, target] + EDGE_WEIGHT_EPSILON)
                    )

        num_edges = len(edge_list)
        if num_edges == 0:
            return {"nodes": nodes, "edges": [], "edge_features": []}

        sum_raw_weights = sum(raw_weights)
        normalized_weights = [
            weight / (sum_raw_weights + EDGE_WEIGHT_EPSILON)
            for weight in raw_weights
        ]
        weight_by_edge = {
            edge: normalized_weights[index] for index, edge in enumerate(edge_list)
        }

        degrees = np.zeros(num_nodes)
        node_weight_sums = np.zeros(num_nodes)
        for source, target in edge_list:
            weight = weight_by_edge[(source, target)]
            degrees[source] += 1
            degrees[target] += 1
            node_weight_sums[source] += weight
            node_weight_sums[target] += weight

        edge_features = []
        for source, target in edge_list:
            weight = weight_by_edge[(source, target)]
            source_degree = degrees[source] / (num_edges + EDGE_WEIGHT_EPSILON)
            target_degree = degrees[target] / (num_edges + EDGE_WEIGHT_EPSILON)
            source_proportion = weight / (
                node_weight_sums[source] + EDGE_WEIGHT_EPSILON
            )
            target_proportion = weight / (
                node_weight_sums[target] + EDGE_WEIGHT_EPSILON
            )
            edge_features.append(
                np.array(
                    [
                        source_degree,
                        source_proportion,
                        target_degree,
                        target_proportion,
                    ],
                    dtype=float,
                )
            )

        return {"nodes": nodes, "edges": edge_list, "edge_features": edge_features}

    def _kG_unnormalized(self, i: int, j: int) -> float:
        """Return the unnormalized graph kernel between two bags."""
        graph_i = self.graphs[i]
        graph_j = self.graphs[j]

        nodes_i = graph_i["nodes"]
        nodes_j = graph_j["nodes"]

        k_node_sum = 0.0
        if nodes_i.shape[0] > 0 and nodes_j.shape[0] > 0:
            dist_sq_nodes = np.sum(
                (nodes_i[:, np.newaxis, :] - nodes_j[np.newaxis, :, :]) ** 2,
                axis=-1,
            )
            k_node_sum = np.sum(np.exp(-self.gamma_node * dist_sq_nodes))

        k_edge_sum = 0.0
        edges_i = graph_i["edge_features"]
        edges_j = graph_j["edge_features"]

        if edges_i and edges_j:
            edges_i_arr = np.array(edges_i)
            edges_j_arr = np.array(edges_j)

            for start_idx in range(0, len(edges_i_arr), EDGE_KERNEL_CHUNK_SIZE):
                end_idx = min(
                    start_idx + EDGE_KERNEL_CHUNK_SIZE,
                    len(edges_i_arr),
                )
                chunk_i = edges_i_arr[start_idx:end_idx]
                dist_sq_edges = np.sum(
                    (
                        chunk_i[:, np.newaxis, :]
                        - edges_j_arr[np.newaxis, :, :]
                    )
                    ** 2,
                    axis=-1,
                )
                k_edge_sum += np.sum(np.exp(-self.gamma_edge * dist_sq_edges))

        return k_node_sum + k_edge_sum

    def compute_kernel_matrix(self) -> np.ndarray:
        """Compute the full symmetric MIGraph kernel matrix."""
        num_bags = self.num_bags
        kernel_matrix = np.zeros((num_bags, num_bags))

        for row in range(num_bags):
            for col in range(row, num_bags):
                kernel_matrix[row, col] = self.sim_between_bags(row, col)
                if row != col:
                    kernel_matrix[col, row] = kernel_matrix[row, col]

        return kernel_matrix

    def sim_between_bags(self, i: int, j: int) -> float:
        """Return the normalized MIGraph similarity between two bags."""
        kg = self._kG_unnormalized(i, j)
        return kg / math.sqrt(
            self.kG_self[i] * self.kG_self[j] + KERNEL_NORMALIZATION_EPSILON
        )

    def set_gamma(self, new_gamma_node: float, new_gamma_edge: float) -> None:
        """Update node/edge gamma values and recompute the kernel matrix."""
        self.gamma_node = new_gamma_node
        self.gamma_edge = new_gamma_edge
        self.kG_self = [self._kG_unnormalized(i, i) for i in range(self.num_bags)]
        self.kernel_matrix = self.compute_kernel_matrix()

    def get_mapping(self):
        """Yield train/test precomputed-kernel matrices for k-fold evaluation."""
        sim_matrix = self.kernel_matrix
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
                sim = self.kernel_matrix[indices[row], indices[col]]
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


__all__ = ["MIGraph"]
