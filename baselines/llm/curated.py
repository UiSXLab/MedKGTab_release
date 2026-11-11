from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .support import complete, messages


def training_table(x_train, y_train, label):
    real = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    if label not in real:
        raise ValueError(
            f"Curated LLM requires the original classification label column '{label}' "
            "in X_train.csv or Y_train.csv; set --curation-label to its name."
        )
    classes = np.unique(real[label].to_numpy())
    if len(classes) < 2 or not np.equal(classes, np.floor(classes)).all():
        raise ValueError("Curated LLM requires a discrete classification label with at least two classes.")
    return real, classes


def preview(x_train, y_train, query, label="label"):
    real, _ = training_table(x_train, y_train, label)
    return messages(real, query.iloc[:1])


def predict(client, model, x_train, y_train, query, *, label="label",
            temperature=0.7, max_tokens=4096, retries_limit=3, nest=600,
            x_threshold=0.2, y_threshold=0.2, concurrency=100):
    from .curation import data_centric_curation

    real, classes = training_table(x_train, y_train, label)
    targets = list(y_train.columns)
    training_labels = np.searchsorted(classes, real[label].to_numpy())
    pending = list(range(len(query)))
    result = np.empty((len(query), len(targets)), dtype=float)
    retries, history = 0, []
    while pending and retries < retries_limit:
        current = query.iloc[pending].reset_index(drop=True)

        def generate(position):
            return complete(client, model, real, current.iloc[[position]], targets,
                            temperature=temperature, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            generated = pd.concat(list(pool.map(generate, range(len(current)))), ignore_index=True)
        result[pending] = generated[targets].to_numpy(dtype=float)
        check_labels = generated[label].to_numpy()
        valid = np.isin(check_labels, classes)
        invalid = np.flatnonzero(~valid).tolist()
        easy, ambiguous, hard = [], [], []
        if valid.any():
            valid_positions = np.flatnonzero(valid)
            groups = data_centric_curation(
                real.drop(columns=[label]), training_labels,
                generated.loc[valid, real.columns].drop(columns=[label]),
                np.searchsorted(classes, check_labels[valid]),
                curation_metric="aleatoric", retrain=False, nest=nest,
                curation_xthresh=x_threshold, curation_ythresh=y_threshold,
            )
            easy, ambiguous, hard = [
                valid_positions[np.asarray(group, dtype=int)].tolist() for group in groups[:3]
            ]
        retry_positions = sorted(set(invalid + ambiguous + hard))
        next_pending = [pending[position] for position in retry_positions]
        retries = retries + 1 if len(next_pending) == len(pending) else 0
        history.append({
            "remaining_before": len(pending), "remaining_after": len(next_pending),
            "retries": retries, "curated": len(easy), "ambiguous": len(ambiguous),
            "unlearnable": len(hard), "invalid_label": len(invalid),
        })
        print(f"curated-llm: {len(next_pending)} rows remain; retries={retries}/{retries_limit}")
        pending = next_pending
    return result, {"round_history": history, "uncurated_row_ids": pending}
