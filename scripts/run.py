from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loop import ExperimentConfig, train_one


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Esegue e aggrega una matrice di esperimenti.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True, choices=["binary", "multiclass", "multilabel"])
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--encoder", default="bert-base-uncased")
    p.add_argument("--models", nargs="+", default=["mil", "gnn", "whole_text"],
                   choices=["mil", "gnn", "whole_text"])
    p.add_argument("--granularities", type=int, nargs="+", default=[128])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--edge-modes", nargs="+", default=["sequential"],
                   choices=["sequential", "knn", "both"],
                   help="Solo per la GNN: quali costruzioni di archi ablare.")
    # iperparametri passati a tutte le celle
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--no-balance", action="store_true", help="disattiva pos_weight/class_weight")
    p.add_argument("--results-root", default="results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cv", action="store_true",
                   help="usa Monte Carlo cross-validation (split generato per fold) "
                        "invece dello split ufficiale")
    return p.parse_args()


def build_configs(args) -> list[ExperimentConfig]:
    base = ExperimentConfig(
        dataset=args.dataset, model="mil", task=args.task, num_classes=args.num_classes,
        encoder=args.encoder, max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, max_length=args.max_length,
        balance=not args.no_balance, results_root=args.results_root,
        cv=args.cv,
    )

    configs: list[ExperimentConfig] = []
    for model, chunk, seed in product(args.models, args.granularities, args.seeds):
        if model == "gnn":
            for edge_mode in args.edge_modes:
                configs.append(replace(base, model=model, chunk_size=chunk,
                                       overlap=chunk // 8, seed=seed, edge_mode=edge_mode))
        else:
            configs.append(replace(base, model=model, chunk_size=chunk,
                                   overlap=chunk // 8, seed=seed))
    return configs

def result_exists(cfg: ExperimentConfig) -> bool:
    return (Path(cfg.results_root) / cfg.run_name() / "result.json").exists()

def load_result(cfg: ExperimentConfig) -> dict:
    return json.loads((Path(cfg.results_root) / cfg.run_name() / "result.json").read_text())

def aggregate(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in results:
        # rimuovi '_seed{n}' dalla chiave di raggruppamento
        name = r["run_name"]
        key = name.rsplit("_seed", 1)[0] + (name.split("_seed", 1)[1][2:] if "_seed" in name else "")
        # tiene un eventuale suffisso (es. _sequential della gnn) dopo il seed
        key = _group_key(name)
        groups.setdefault(key, []).append(r)

    summary: dict[str, dict] = {}
    for key, runs in sorted(groups.items()):
        metric_names = runs[0]["test_metrics"].keys()
        agg = {}
        for m in metric_names:
            vals = [run["test_metrics"][m] for run in runs]
            agg[m] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        summary[key] = agg
    return summary


def _group_key(run_name: str) -> str:
    parts = run_name.split("_")
    return "_".join(p for p in parts if not p.startswith("seed"))


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    print(f"[run] matrice: {len(configs)} celle da eseguire.\n")

    all_results: list[dict] = []
    for i, cfg in enumerate(configs, 1):
        tag = f"[{i}/{len(configs)}] {cfg.run_name()}"
        if result_exists(cfg) and not args.overwrite:
            print(f"{tag}  ->  gia presente, salto.")
            all_results.append(load_result(cfg))
            continue
        print(f"{tag}  ->  in esecuzione...")
        result = train_one(cfg)
        all_results.append(result)

    # aggregazione e stampa del riassunto
    summary = aggregate(all_results)
    print("RIASSUNTO (media +- std sui seed)")
    for group, metrics in summary.items():
        print(f"\n{group}   (n={list(metrics.values())[0]['n']} seed)")
        for m, stats in metrics.items():
            if m.endswith("loss"):
                continue
            print(f"  {m:16s} {stats['mean']:.3f} ± {stats['std']:.3f}")

    # salva il riassunto aggregato
    out = Path(args.results_root) / f"summary_{args.dataset}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[run] riassunto salvato in {out}")


if __name__ == "__main__":
    main()