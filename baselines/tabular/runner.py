import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

METHODS = ("xgboost", "ft-transformer", "danets", "switchtab", "modernnca")


def predict(method, x_train, y_train, x_test, *, seed=0, epochs=600,
            device="cpu", batch_size=1024, val_fraction=0, checkpoint_dir=None):
    x = x_train.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    train = np.arange(len(x))
    validation = train
    if val_fraction:
        train, validation = train_test_split(train, test_size=val_fraction, random_state=seed)
    if len(train) < 2:
        raise ValueError("At least two training rows are needed after the validation split.")

    if method == "xgboost":
        from xgboost import XGBRegressor

        columns = []
        for target in range(y.shape[1]):
            model = XGBRegressor(
                objective="reg:squarederror", n_estimators=600, learning_rate=0.05,
                max_depth=6, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0, reg_lambda=1, early_stopping_rounds=50,
                tree_method="hist", random_state=seed, n_jobs=-1,
                device=device if device.startswith("cuda") else "cpu",
            )
            model.fit(x[train], y[train, target],
                      eval_set=[(x[validation], y[validation, target])], verbose=False)
            columns.append(model.predict(x_test.to_numpy(dtype=float)))
        return np.column_stack(columns)

    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    import torch

    talent_dir = Path(__file__).resolve().parent / "talent"
    sys.path.insert(0, str(talent_dir))
    from TALENT.model.methods.danets import DANetsMethod
    from TALENT.model.methods.ftt import FTTMethod
    from TALENT.model.methods.modernNCA import ModernNCAMethod
    from TALENT.model.methods.switchtab import SwitchTabMethod
    from TALENT.model.utils import set_seeds

    methods = {
        "ft-transformer": (FTTMethod, "indices"),
        "danets": (DANetsMethod, "ordinal"),
        "switchtab": (SwitchTabMethod, "ordinal"),
        "modernnca": (ModernNCAMethod, "tabr_ohe"),
    }
    configs = json.loads((talent_dir / "config.json").read_text())
    method_class, cat_policy = methods[method]
    checkpoint_dir = Path(checkpoint_dir or Path("outputs") / method / "checkpoints")
    info = {"task_type": "regression", "n_num_features": x.shape[1], "n_cat_features": 0}
    columns = []
    for target in range(y.shape[1]):
        # Use the original TALENT methods for preprocessing, fitting and inference.
        # The experiment's data_label_process keeps targets on their original scale.
        config = copy.deepcopy(configs[method])
        config["training"]["n_bins"] = 2
        save_path = checkpoint_dir / f"target_{target}" / f"seed_{seed}"
        save_path.mkdir(parents=True, exist_ok=True)
        args = argparse.Namespace(
            config=config, seed=seed, max_epoch=epochs, batch_size=batch_size,
            normalization="standard", num_nan_policy="mean", cat_nan_policy="new",
            cat_policy=cat_policy, num_policy="none", use_float=False,
            save_path=str(save_path), device=torch.device(device),
        )
        set_seeds(seed)
        trainer = method_class(args, True)
        trainer.args.device = torch.device(device)
        data = (
            {"train": x[train].copy(), "val": x[validation].copy()},
            None,
            {"train": y[train, target].copy(), "val": y[validation, target].copy()},
        )
        trainer.fit(data, info)
        # Test labels are only used by TALENT's reporting; never load Y_test here.
        test_data = (
            {"test": x_test.to_numpy(dtype=float)}, None,
            {"test": np.zeros(len(x_test), dtype=float)},
        )
        _, _, _, prediction = trainer.predict(test_data, info, model_name="best-val")
        prediction = prediction.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{method} produced nonfinite predictions.")
        columns.append(prediction)
    return np.column_stack(columns)
