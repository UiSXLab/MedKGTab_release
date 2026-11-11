import argparse
from pathlib import Path

import numpy as np

from common.data import load_data, save_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=[
        "medkgtab", "tabpfn", "xgboost", "ft-transformer", "danets", "switchtab", "modernnca"
    ])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("tabpfn-v2.5-regressor.ckpt"))
    parser.add_argument("--graph", type=Path, help="Raw adjacency, ordered as [X columns, Y columns].")
    parser.add_argument("--normalized-graph", action="store_true", help="Graph is already normalized.")
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0 <= args.val_fraction < 1:
        parser.error("epochs >= 1, batch-size >= 2 and 0 <= val-fraction < 1 are required.")
    if not np.isfinite(args.alpha):
        parser.error("alpha must be finite.")
    if args.method == "medkgtab" and args.graph is None:
        parser.error("MedKGTab requires --graph.")
    x_train, y_train, x_test = load_data(args.data)
    if args.method in ("medkgtab", "tabpfn"):
        from medkgtab.model import normalize_graph, predict
        graph = None
        if args.method == "medkgtab":
            graph = np.load(args.graph, allow_pickle=False)
            if not np.isfinite(graph).all() or (graph < 0).any():
                parser.error("graph must contain finite nonnegative values.")
            if not args.normalized_graph:
                graph = normalize_graph(graph)
        prediction = predict(x_train, y_train, x_test, graph=graph, alpha=args.alpha,
                             checkpoint=args.checkpoint, device=args.device, seed=args.seed)
    else:
        from baselines.tabular import predict
        prediction = predict(args.method, x_train, y_train, x_test, seed=args.seed,
                             epochs=args.epochs, device=args.device,
                             batch_size=args.batch_size, val_fraction=args.val_fraction,
                             checkpoint_dir=(args.output or Path("outputs") / args.method)
                             / "checkpoints")
    save_predictions(prediction, y_train.columns, args.data,
                     args.output or Path("outputs") / args.method,
                     method=args.method, seed=args.seed,
                     **({"alpha": args.alpha} if args.method == "medkgtab" else {}))


if __name__ == "__main__":
    main()
