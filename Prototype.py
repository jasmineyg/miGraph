"""Core data container utilities for multiple-instance learning datasets."""

from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.io as scio

PathLike = Union[str, Path]


def load_mat_bags(path: PathLike) -> np.ndarray:
    """Load MIL bags from a MATLAB file containing a ``data`` variable."""
    data = scio.loadmat(path)
    if "data" not in data:
        raise ValueError("MAT file must contain a 'data' variable.")
    return data["data"]


def _scalar_label(value: Any) -> Any:
    """Return a scalar bag label from MATLAB/Python label containers."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError("Bag label must be a scalar value.")
    return arr.reshape(-1)[0]


class MIL:
    """Base container for MIL bags and commonly used bag/instance indexes."""

    def __init__(self, data: Any, para_has_ins_label: bool = True) -> None:
        self.data_name = ""
        self.bags = []
        self.num_bags = 0
        self.num_classes = 0
        self.bags_size = []
        self.bags_label = []
        self.num_ins = 0
        self.dimensions = 0
        self.ins = []
        self.ins_idx = []
        self.ins_lab = []
        self.has_ins_label = para_has_ins_label
        self._initialize(data)

    def _initialize(self, data: Any) -> None:
        if isinstance(data, (str, Path)):
            path = Path(data)
            self.bags = load_mat_bags(path)
            self.data_name = path.stem
        else:
            self.bags = np.asarray(data, dtype=object)
            self.data_name = "in_memory"

        self.num_bags = int(np.shape(self.bags)[0])
        if self.num_bags == 0:
            raise ValueError("MIL data must contain at least one bag.")

        self.bags_size = np.zeros(self.num_bags, dtype=int)
        self.bags_label = np.zeros(self.num_bags, dtype=int)

        first_non_empty = None
        for bag_index in range(self.num_bags):
            bag = np.asarray(self.bags[bag_index, 0])
            if bag.ndim != 2:
                raise ValueError("Each bag must be a 2-D instance matrix.")

            self.bags[bag_index, 0] = bag
            self.bags_size[bag_index] = bag.shape[0]
            self.bags_label[bag_index] = _scalar_label(self.bags[bag_index, 1])

            if first_non_empty is None and bag.shape[0] > 0:
                first_non_empty = bag

        if first_non_empty is None:
            raise ValueError("MIL data must contain at least one instance.")

        self.num_ins = int(np.sum(self.bags_size))
        self.dimensions = first_non_empty.shape[1]
        if self.has_ins_label:
            self.dimensions -= 1
        if self.dimensions <= 0:
            raise ValueError("Instance feature dimension must be positive.")

        self.num_classes = len(set(self.bags_label.tolist()))
        self.ins = np.zeros((self.num_ins, self.dimensions))
        self.ins_idx = np.zeros(self.num_bags + 1, dtype=int)
        self.ins_lab = np.zeros(self.num_ins, dtype=int)

        for bag_index in range(self.num_bags):
            self.ins_idx[bag_index + 1] = (
                self.bags_size[bag_index] + self.ins_idx[bag_index]
            )
            start = self.ins_idx[bag_index]
            end = self.ins_idx[bag_index + 1]
            self.ins[start:end] = self.bags[bag_index, 0][:, : self.dimensions]
            self.ins_lab[start:end] = bag_index

    def get_info(self) -> None:
        """Print a compact summary of the loaded MIL dataset."""
        preview_size = min(5, self.num_bags)
        print(
            "{} information:".format(self.data_name),
            "\nNumber bags:",
            self.num_bags,
            "\nNumber classes:",
            self.num_classes,
            "\nBag size:",
            self.bags_size[:preview_size],
            "...",
            "\nBag label:",
            self.bags_label[:preview_size],
            "...",
            "\nMaximum bag size:",
            np.max(self.bags_size),
            "\nMinimum bag size:",
            np.min(self.bags_size),
            "\nNumber instances:",
            self.num_ins,
            "\nInstance dimensions:",
            self.dimensions,
            "\nInstance index:",
            self.ins_idx[:preview_size],
            "...",
            "\nInstance bag index:",
            self.ins_lab[:preview_size],
            "...",
            "\nInstance labels provided?",
            self.has_ins_label,
        )

    def get_index(self, para_k: int = 10):
        """Return randomized train/test bag indexes for k-fold evaluation."""
        if para_k <= 1:
            raise ValueError("para_k must be greater than 1.")
        if para_k > self.num_bags:
            raise ValueError("para_k cannot exceed the number of bags.")

        temp_rand_idx = np.random.permutation(self.num_bags)
        folds = np.array_split(temp_rand_idx, para_k)
        ret_tr_idx = {}
        ret_te_idx = {}

        for fold_index, test_idx in enumerate(folds):
            train_idx = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            ret_tr_idx[fold_index] = train_idx.tolist()
            ret_te_idx[fold_index] = test_idx.tolist()
        return ret_tr_idx, ret_te_idx

    def get_ins(self, idx) -> np.ndarray:
        """Return the concatenated instances for selected bag indexes."""
        temp_num_ins = np.sum(self.bags_size[idx])
        ret_ins = np.zeros((temp_num_ins, self.dimensions))

        temp_count = 0
        for bag_index in idx:
            temp_size = self.bags_size[bag_index]
            ret_ins[temp_count: temp_count + temp_size] = self.bags[
                bag_index, 0
            ][:, : self.dimensions]
            temp_count += temp_size

        return ret_ins


__all__ = ["MIL", "load_mat_bags"]
