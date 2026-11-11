import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def load_data(directory):
    directory = Path(directory)
    frames = [
        pd.read_csv(directory / name)
        for name in ("X_train.csv", "Y_train.csv", "X_test.csv")
    ]
    x_train, y_train, x_test = frames
    if len(x_train) != len(y_train) or len(x_train) < 2:
        raise ValueError("Training X/Y must have matching rows and at least two records.")
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("X_train/X_test must have identical columns in the same order.")
    for frame in frames:
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("CSV files must be nonempty with unique column names.")
        if any(str(name).startswith("Unnamed:") for name in frame.columns):
            raise ValueError("Remove exported CSV index columns (Unnamed: ...).")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("CSV files must contain finite numeric values only.")
    return x_train, y_train, x_test


def save_predictions(prediction, target_columns, data_dir, output_dir, **metadata):
    output_dir = Path(output_dir)
    frame = pd.DataFrame(prediction, columns=target_columns)
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be nonempty and finite.")
    truth_path = Path(data_dir) / "Y_test.csv"
    if truth_path.is_file():
        truth = pd.read_csv(truth_path)
        if list(truth.columns) != list(target_columns) or truth.shape != frame.shape:
            raise ValueError("Y_test must match the predicted rows and target columns.")
        actual = truth.to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError("Y_test must contain finite numbers.")
        error = actual - frame.to_numpy(dtype=float)
        sample_mae = np.abs(error).mean(axis=1)
        metadata.update(
            mse=float(np.mean(error**2)),
            r2=(float(r2_score(actual.ravel(), frame.to_numpy().ravel()))
                if actual.size >= 2 else None),
            medae=float(np.median(sample_mae)),
            mae_p95=float(np.quantile(sample_mae, 0.95)),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(frame)} rows to {output_dir / 'predictions.csv'}")
