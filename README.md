# miGraph / MIGraph

This repository reproduces the miGraph / MIGraph algorithms from the paper
*Multi-instance learning by treating instances as non-I.I.D. samples*.

Only the core algorithm code is tracked. Data preprocessing scripts, timing
scripts, IDE files, generated caches, and dataset files are intentionally left
out of git.

## Requirements

```powershell
pip install -r requirements.txt
```

## Run

The code expects MIL data in the usual MATLAB format, where the `.mat` file
contains a `data` variable. Each row is a bag: `data[i, 0]` is the instance
matrix and `data[i, 1]` is the bag label. Instance labels, when present, should
be stored in the last column of each instance matrix.

Build the miGraph precomputed kernel folds:

```powershell
python -c "from miGraph import miGraph; model = miGraph('path/to/data.mat'); train_K, train_y, test_K, test_y, _ = next(model.get_mapping()); print(train_K.shape, test_K.shape)"
```

Build the MIGraph precomputed kernel folds:

```powershell
python -c "from MIGraph_explicit import MIGraph; model = MIGraph('path/to/data.mat'); train_K, train_y, test_K, test_y, _ = next(model.get_mapping()); print(train_K.shape, test_K.shape)"
```

The returned matrices are precomputed kernels and can be passed to a classifier
that supports precomputed kernels.
