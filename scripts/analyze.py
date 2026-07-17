"""
analyze.py — Aggrega i risultati e produce tabelle, test di significativita e
figure per la relazione.

SEPARAZIONE UFFICIALE vs CROSS-VALIDATION:
  I run con Monte Carlo CV (config['cv']==True, cartelle '*_cv') e quelli con
  split ufficiale sono due PROTOCOLLI diversi e non vanno mai mescolati. Per
  ogni dataset che ha entrambi (daic, imcs21, mtsamples) si producono percio'
  DUE analisi separate:
     - "<dataset> [ufficiale]"  -> solo run a split ufficiale, confrontati fra loro
     - "<dataset> [cv]"         -> solo run in Monte Carlo CV, confrontati fra loro
  I confronti a coppie e le figure restano SEMPRE dentro lo stesso protocollo.
  ECtHR ha solo lo split ufficiale -> una sola analisi.

Legge results/**/result.json (con 'predictions' e 'timing') e per ogni blocco
produce: tabella (CSV), McNemar sulle predizioni per-documento, figure.

Uso:
    python scripts/analyze.py
    python scripts/analyze.py --datasets daic ecthr
    python scripts/analyze.py --metric test/auprc
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ------------------------------------------------------------- caricamento ----

def load_all_results(results_root: Path) -> list[dict]:
    results = []
    for path in sorted(results_root.glob("*/result.json")):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[analyze] salto {path}: {e}")
    return results


def is_cv(cfg: dict) -> bool:
    """True se il run e' in Monte Carlo cross-validation."""
    return bool(cfg.get("cv", False))


def protocol_of(cfg: dict) -> str:
    """Il protocollo del run: 'cv' o 'ufficiale'. E' la chiave che separa i
    due mondi: non si confrontano mai fra loro."""
    return "cv" if is_cv(cfg) else "ufficiale"


def group_key(cfg: dict) -> str:
    """Identifica un setup a meno del seed, DENTRO un protocollo.
    Include '_cv' per i run in cross-validation, cosi non collassano con gli
    ufficiali nemmeno se dataset/modello/granularita coincidono."""
    base = f"{cfg['dataset']}_{cfg['model']}_g{cfg['chunk_size']}"
    if cfg["model"] == "gnn":
        base += f"_{cfg.get('edge_mode', 'sequential')}"
    if is_cv(cfg):
        base += "_cv"
    return base


def build_groups(results: list[dict]) -> dict[str, dict]:
    """Raggruppa per setup (a meno del seed). Ogni gruppo conserva, per seed,
    metriche + predizioni + timing, e sa a quale (dataset, protocollo) appartiene."""
    groups: dict[str, dict] = {}
    for r in results:
        cfg = r["config"]
        key = group_key(cfg)
        g = groups.setdefault(key, {
            "dataset": cfg["dataset"],
            "protocol": protocol_of(cfg),        # 'cv' | 'ufficiale'
            "model": cfg["model"],
            "chunk_size": cfg["chunk_size"],
            "edge_mode": cfg.get("edge_mode") if cfg["model"] == "gnn" else None,
            "by_seed": defaultdict(dict),
            "preds_by_seed": {},
            "timing_by_seed": {},
        })
        seed = cfg["seed"]
        for m, v in r["test_metrics"].items():
            g["by_seed"][seed][m] = v
        if r.get("predictions"):
            g["preds_by_seed"][seed] = {
                p["doc_id"]: (p["label"], p["probs"]) for p in r["predictions"]
            }
        if r.get("timing"):
            g["timing_by_seed"][seed] = r["timing"]
    return groups


def metric_values(group: dict, metric: str) -> dict[int, float]:
    return {s: mm[metric] for s, mm in group["by_seed"].items() if metric in mm}


def all_metrics(group: dict) -> list[str]:
    names = set()
    for mm in group["by_seed"].values():
        names.update(mm.keys())
    return sorted(n for n in names if not n.endswith("loss"))


def primary_metric(metrics: list[str]) -> str:
    for cand in ("test/f1_macro", "test/f1", "test/auprc", "test/auroc"):
        if cand in metrics:
            return cand
    return metrics[0] if metrics else ""


def aggregate(group: dict, metric: str) -> tuple[float, float, int]:
    vals = list(metric_values(group, metric).values())
    if not vals:
        return float("nan"), 0.0, 0
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def mean_timing(group: dict, field: str) -> float | None:
    vals = [t[field] for t in group["timing_by_seed"].values()
            if t.get(field) is not None]
    return statistics.mean(vals) if vals else None


# --------------------------------------------------------------- tabelle ----

def print_table(title: str, keys: list[str], groups: dict[str, dict], out_dir: Path,
                csv_name: str) -> None:
    if not keys:
        return
    metrics = sorted(set().union(*(all_metrics(groups[k]) for k in keys)))
    print(f"\n{'='*74}\n{title}  —  media ± std sui seed\n{'='*74}")
    print(f"{'setup':32s} " + " ".join(f"{m.replace('test/',''):>11s}" for m in metrics)
          + f" {'infer.ms':>9s}")
    rows_csv = []
    for k in keys:
        n = max((len(metric_values(groups[k], m)) for m in metrics), default=0)
        cells, csv_row = [], {"setup": k, "n_seed": n}
        for m in metrics:
            mean, std, _ = aggregate(groups[k], m)
            cells.append(f"{mean:.3f}±{std:.3f}")
            csv_row[m] = f"{mean:.4f}"; csv_row[m + "_std"] = f"{std:.4f}"
        ms = mean_timing(groups[k], "inference_ms_per_doc")
        params = mean_timing(groups[k], "trainable_params")
        csv_row["inference_ms_per_doc"] = f"{ms:.3f}" if ms else ""
        csv_row["trainable_params"] = int(params) if params else ""
        ms_str = f"{ms:>9.2f}" if ms else " " * 9
        print(f"{k:32s} " + " ".join(f"{c:>11s}" for c in cells) + f" {ms_str}  (n={n})")
        rows_csv.append(csv_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name
    fieldnames = (["setup", "n_seed"]
                  + [c for m in metrics for c in (m, m + "_std")]
                  + ["inference_ms_per_doc", "trainable_params"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows_csv)
    print(f"[analyze] tabella -> {csv_path}")


# ---------------------------------------------------- McNemar / significativita ----

def _binarize(probs: list[float], task_multilabel: bool) -> object:
    if task_multilabel:
        return tuple(1 if p >= 0.5 else 0 for p in probs)
    if len(probs) == 1:
        return int(probs[0] >= 0.5)
    return int(max(range(len(probs)), key=lambda i: probs[i]))


def _correct_map(preds: dict, task_multilabel: bool) -> dict[str, bool]:
    out = {}
    for doc_id, (label, probs) in preds.items():
        pred = _binarize(probs, task_multilabel)
        if task_multilabel:
            gold = tuple(1 if i in label else 0 for i in range(len(probs)))
            out[doc_id] = (pred == gold)
        else:
            out[doc_id] = (pred == label)
    return out


def mcnemar_pair(group_a: dict, group_b: dict, task_multilabel: bool) -> dict | None:
    shared_seeds = sorted(set(group_a["preds_by_seed"]) & set(group_b["preds_by_seed"]))
    if not shared_seeds:
        return None
    b, c, n = 0, 0, 0
    for seed in shared_seeds:
        corr_a = _correct_map(group_a["preds_by_seed"][seed], task_multilabel)
        corr_b = _correct_map(group_b["preds_by_seed"][seed], task_multilabel)
        for doc_id in set(corr_a) & set(corr_b):
            ca, cb = corr_a[doc_id], corr_b[doc_id]
            if ca and not cb: b += 1
            elif cb and not ca: c += 1
            n += 1
    if b + c == 0:
        return {"b": 0, "c": 0, "n": n, "pvalue": 1.0}
    if _HAS_SCIPY:
        pvalue = _stats.binomtest(min(b, c), b + c, 0.5).pvalue
    else:
        pvalue = float("nan")
    return {"b": b, "c": c, "n": n, "pvalue": pvalue}


def significance(title: str, keys: list[str], groups: dict[str, dict], metric: str) -> None:
    if len(keys) < 2:
        return
    task_ml = any("f1_micro" in all_metrics(groups[k]) and groups[k]["model"] != "gnn"
                  for k in keys) or any(
                  "f1_micro" in all_metrics(groups[k]) for k in keys)
    print(f"\n{title} — confronto a coppie (stesso protocollo)")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            la = a.replace(f"{groups[a]['dataset']}_", "")
            lb = b.replace(f"{groups[b]['dataset']}_", "")
            ma, _, _ = aggregate(groups[a], metric)
            mb, _, _ = aggregate(groups[b], metric)
            diff = ma - mb
            mc = mcnemar_pair(groups[a], groups[b], task_ml)
            if mc is not None:
                print(f"  {la} vs {lb}: Δ{metric.replace('test/','')}={diff:+.3f} | "
                      f"McNemar b={mc['b']} c={mc['c']} p={mc['pvalue']:.4f} (n={mc['n']})")
            else:
                va, vb = metric_values(groups[a], metric), metric_values(groups[b], metric)
                shared = sorted(set(va) & set(vb))
                if len(shared) >= 2 and _HAS_SCIPY:
                    p = _stats.ttest_rel([va[s] for s in shared], [vb[s] for s in shared]).pvalue
                    print(f"  {la} vs {lb}: Δ={diff:+.3f} | t-test seed p={p:.3f} (n={len(shared)})")
                else:
                    print(f"  {la} vs {lb}: Δ={diff:+.3f} | test non calcolabile")


# --------------------------------------------------------------- figure ----

def plot_bars(title: str, keys: list[str], groups: dict[str, dict], metric: str,
              out_dir: Path, fig_name: str) -> None:
    if not _HAS_MPL or not keys:
        return
    means, stds, labels = [], [], []
    for k in keys:
        mean, std, n = aggregate(groups[k], metric)
        if n == 0: continue
        means.append(mean); stds.append(std)
        labels.append(k.replace(f"{groups[k]['dataset']}_", ""))
    if not means: return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.5))
    ax.bar(range(len(means)), means, yerr=stds, capsize=5,
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric.replace("test/", "")); ax.set_ylim(0, 1)
    ax.set_title(f"{title} — {metric.replace('test/','')} (media ± std)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[analyze] figura -> {path}")


def plot_acc_vs_cost(title: str, keys: list[str], groups: dict[str, dict], metric: str,
                     out_dir: Path, fig_name: str) -> None:
    if not _HAS_MPL or not keys:
        return
    xs, ys, labels = [], [], []
    for k in keys:
        mean, _, n = aggregate(groups[k], metric)
        ms = mean_timing(groups[k], "inference_ms_per_doc")
        if n == 0 or ms is None: continue
        xs.append(ms); ys.append(mean); labels.append(k.replace(f"{groups[k]['dataset']}_", ""))
    if not xs: return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(xs, ys, s=80, color="#C44E52", edgecolor="black", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("latenza inferenza per documento (ms, log)")
    ax.set_ylabel(metric.replace("test/", ""))
    ax.set_title(f"{title} — qualità vs costo")
    ax.grid(alpha=0.3); fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[analyze] figura -> {path}")

def plot_boxplot(title: str, keys: list[str], groups: dict[str, dict], metric: str,
                 out_dir: Path, fig_name: str) -> None:
    """Boxplot della metrica primaria: una scatola per modello, costruita sui
    valori dei singoli seed/fold. Mostra la DISTRIBUZIONE (mediana, quartili,
    dispersione, outlier), non solo media±std -> piu' onesto sui dati rumorosi
    e piu' informativo sui blocchi CV (10 fold)."""
    if not _HAS_MPL or not keys:
        return

    data, labels, ns = [], [], []
    for k in keys:
        vals = list(metric_values(groups[k], metric).values())
        if not vals:
            continue
        data.append(vals)
        labels.append(k.replace(f"{groups[k]['dataset']}_", ""))
        ns.append(len(vals))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))

    bp = ax.boxplot(
        data,
        tick_labels=labels,          # per matplotlib<3.9 usare labels=labels
        showmeans=True,              # mostra anche la MEDIA (triangolo) oltre alla mediana
        meanprops=dict(marker="^", markerfacecolor="white",
                       markeredgecolor="black", markersize=7),
        medianprops=dict(color="#C44E52", linewidth=2),   # mediana in risalto
        boxprops=dict(facecolor="#4C72B0", alpha=0.6, edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=5, markerfacecolor="gray", alpha=0.6),
        patch_artist=True,           # necessario per colorare i box
    )

    # sovrappongo i singoli punti (seed/fold), leggermente sparpagliati in x,
    # cosi si vede QUANTI sono e dove cadono davvero
    import numpy as np
    for i, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=18, color="black", alpha=0.5, zorder=3)

    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("test/", ""))
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} — {metric.replace('test/','')} (distribuzione sui fold)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([bp["medians"][0], bp["means"][0]], ["mediana", "media"],
              loc="lower right", fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analyze] figura -> {path}")

# ----------------------------------------------------------------- main ----

def analyze_block(dataset: str, protocol: str, groups: dict[str, dict],
                  metric_override: str | None, out_dir: Path) -> None:
    """Una analisi completa per un blocco (dataset, protocollo)."""
    keys = sorted(k for k, g in groups.items()
                  if g["dataset"] == dataset and g["protocol"] == protocol)
    if not keys:
        return
    metrics = sorted(set().union(*(all_metrics(groups[k]) for k in keys)))
    metric = metric_override if (metric_override and metric_override in metrics) \
             else primary_metric(metrics)

    title = f"{dataset.upper()} [{protocol}]"
    suffix = f"{dataset}_{protocol}"
    print_table(title, keys, groups, out_dir, csv_name=f"table_{suffix}.csv")
    significance(title, keys, groups, metric)
    plot_bars(title, keys, groups, metric, out_dir, fig_name=f"fig_{suffix}_bars.png")
    plot_boxplot(title, keys, groups, metric, out_dir, fig_name=f"fig_{suffix}_box.png")
    plot_acc_vs_cost(title, keys, groups, metric, out_dir, fig_name=f"fig_{suffix}_cost.png")


def parse_args():
    p = argparse.ArgumentParser(description="Aggrega risultati, test e figure.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out-dir", default="results/analysis")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--metric", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    results = load_all_results(Path(args.results_root))
    if not results:
        print(f"Nessun result.json in {args.results_root}. Hai lanciato run.py?")
        return
    print(f"[analyze] {len(results)} run caricati.")
    if not _HAS_SCIPY:
        print("[analyze] scipy assente: test saltati (pip install scipy).")
    if not _HAS_MPL:
        print("[analyze] matplotlib assente: figure saltate (pip install matplotlib).")

    groups = build_groups(results)
    out_dir = Path(args.out_dir)

    # elenco (dataset, protocollo) presenti, ordinato: prima ufficiale poi cv
    blocks = sorted({(g["dataset"], g["protocol"]) for g in groups.values()},
                    key=lambda t: (t[0], t[1] != "ufficiale"))
    if args.datasets:
        blocks = [b for b in blocks if b[0] in args.datasets]

    for dataset, protocol in blocks:
        analyze_block(dataset, protocol, groups, args.metric, out_dir)

    print(f"\n[analyze] output in {out_dir}/")
    print("NB: blocchi '[ufficiale]' e '[cv]' sono analizzati SEPARATAMENTE:")
    print("    i confronti restano sempre dentro lo stesso protocollo.")


if __name__ == "__main__":
    main()