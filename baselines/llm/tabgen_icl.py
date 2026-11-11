import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .support import complete, messages


def initial_context(real, target, ic_samples):
    # Original first prompt: replacement sampling scaled by unique target values.
    count = min(ic_samples * real[target].nunique(), len(real))
    return real.sample(n=count, replace=True).reset_index(drop=True)


def preview(x_train, y_train, query, seed=0, ic_samples=1):
    np.random.seed(seed)
    real = pd.concat([x_train.reset_index(drop=True), y_train.iloc[:, :1].reset_index(drop=True)], axis=1)
    return messages(initial_context(real, y_train.columns[0], ic_samples), query.iloc[:1])


def predict(client, model, x_train, y_train, query, *, seed=0, ic_samples=1,
            n_samples=5, max_prompt_iters=10, temperature=1.0,
            max_tokens=4096, concurrency=8):
    from .tabgen_utils import compute_distance, get_indices

    np.random.seed(seed)
    random.seed(seed)
    # Precompute the context sequences serially so threading cannot race the RNG.
    # Each query/target has its own generated history, as in the original script.
    def tasks_for_row(row):
        tasks = []
        for target in y_train.columns:
            real = pd.concat([x_train.reset_index(drop=True), y_train[[target]].reset_index(drop=True)], axis=1)
            first = initial_context(real, target, ic_samples)
            later = []
            for _ in range(1, max_prompt_iters):
                column = random.choice(list(real.columns))
                later.append(get_indices(real, column, str(real[column].dtype), max_num=500))
            tasks.append((row, target, real, first, later))
        return tasks

    def generate(task):
        row, target, real, context, later = task
        history = pd.DataFrame(columns=real.columns, dtype=float)
        selections = []
        for prompt_idx in range(max_prompt_iters):
            if prompt_idx:
                distances = compute_distance(
                    real, later[prompt_idx - 1], history, prompt_idx=prompt_idx
                )
                if not np.isfinite(distances).all():
                    raise RuntimeError("TabGen-ICL context distances are not finite.")
                indices = later[prompt_idx - 1][int(np.argmin(distances))]
                context = real.loc[indices].reset_index(drop=True)
            generated = complete(
                client, model, context, query.iloc[[row]], [target],
                temperature=temperature, max_tokens=max_tokens,
            )
            history = pd.concat([history, generated[real.columns]], ignore_index=True)
            selections.append({"iteration": prompt_idx, "context_rows": len(context)})
            if len(history) >= n_samples:
                break
        if len(history) < n_samples:
            raise RuntimeError(
                f"TabGen-ICL generated {len(history)}/{n_samples} samples for "
                f"row {row}, target {target}; increase --max-prompt-iters."
            )
        return row, target, float(history[target].mean()), selections

    result = np.empty((len(query), len(y_train.columns)), dtype=float)
    history = []
    positions = {name: position for position, name in enumerate(y_train.columns)}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row_id in range(len(query)):
            for row, target, value, selections in pool.map(generate, tasks_for_row(row_id)):
                result[row, positions[target]] = value
                history.append({"row_id": row, "target": target, "prompts": selections})
                print(f"tabgen-icl: row {row + 1}/{len(query)}, target {target}")
    return result, {"generation_history": history}
