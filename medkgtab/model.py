import os
import sys
from pathlib import Path

import numpy as np


def normalize_graph(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("The adjacency matrix must be square.")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("The adjacency matrix must contain finite nonnegative weights.")
    adjacency = adjacency + np.eye(len(adjacency), dtype=np.float32)
    scale = 1 / np.sqrt(adjacency.sum(axis=1) + 1e-12)
    return scale[:, None] * adjacency * scale[None, :]


def pool_prior(graph, target, feature_count, group_size):
    groups = (feature_count + group_size - 1) // group_size
    padded_count = groups * group_size
    core = np.zeros((padded_count, padded_count))
    core[:feature_count, :feature_count] = graph[:feature_count, :feature_count]
    core = core.reshape(groups, group_size, groups, group_size).mean(axis=(1, 3))
    column = np.full(padded_count, np.nan)
    row = column.copy()
    column[:feature_count] = graph[:feature_count, target]
    row[:feature_count] = graph[target, :feature_count]
    prior = np.empty((groups + 1, groups + 1), dtype=np.float32)
    prior[:-1, :-1] = core
    prior[:-1, -1] = np.nanmean(column.reshape(groups, group_size), axis=1)
    prior[-1, :-1] = np.nanmean(row.reshape(groups, group_size), axis=1)
    prior[-1, -1] = graph[target, target]
    return prior


def predict(x_train, y_train, x_test, *, checkpoint, device, seed=0, graph=None, alpha=3):
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download the TabPFN regression checkpoint: {checkpoint}")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ["MEDKGTAB_PREPROCESSING"] = "1" if graph is not None else "0"
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tabpfn/src"))
    from tabpfn import TabPFNRegressor

    feature_count = x_train.shape[1]
    if graph is not None and graph.shape != (
        feature_count + y_train.shape[1], feature_count + y_train.shape[1]
    ):
        raise ValueError("Graph order/shape must match [X_train columns, Y_train columns].")
    regressor = TabPFNRegressor(
        model_path=str(checkpoint), n_estimators=1, random_state=seed
    )
    regressor.to(device)
    columns = []
    for position, target in enumerate(y_train.columns):
        regressor.fit(x_train, y_train[target])
        model = regressor.models_[0]
        model.set_bias(None)
        if graph is not None:
            prior = pool_prior(
                graph, feature_count + position, feature_count, int(model.features_per_group)
            )
            dtype = torch.float32 if device == "cpu" else torch.float16
            model.set_bias(alpha * torch.as_tensor(prior, dtype=dtype, device=device))
        try:
            columns.append(regressor.predict(x_test))
        finally:
            model.set_bias(None)
    return np.column_stack(columns)
