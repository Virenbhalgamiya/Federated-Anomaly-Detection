"""Run every experiment across several seeds and write results/ (CSV, RESULTS.md, charts).

    python run_experiments.py                 # 3 seeds, 20 rounds (the reported setting)
    SEEDS=0,1 ROUNDS=5 python run_experiments.py   # quick smoke run

Resumable: finished (seed, partition, method) rows in results/results.csv are skipped.
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.baselines import run_centralized, run_local_only
from src.client import FLConfig
from src.data import load_magic, make_splits, partition
from src.server import run_federated

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", 20))
NOISE_SWEEP = [0.5, 1.0, 2.0]   # IID privacy-utility sweep
PROX_SWEEP = [0.0, 1.0, 10.0]   # non-IID FedProx sweep
DP_NM = 1.0                     # noise level used in the non-IID DP runs
OUT = os.environ.get("OUT", "results/results.csv")  # one file per process when seeds run in parallel


def grid(cfg_base):
    """Yield (partition, method, kwargs) for every configuration of one seed."""
    yield "none", "centralized", None
    yield "iid", "local-only", None
    yield "iid", "fedavg", cfg_base
    for nm in NOISE_SWEEP:
        yield "iid", f"fedavg+dp nm={nm}", cfg_base.__class__(**{**cfg_base.__dict__, "noise_multiplier": nm})
    yield "noniid", "local-only", None
    for mu in PROX_SWEEP:
        yield "noniid", f"fedavg mu={mu:g}", cfg_base.__class__(**{**cfg_base.__dict__, "prox_mu": mu})
        yield "noniid", f"fedavg+dp mu={mu:g}", cfg_base.__class__(
            **{**cfg_base.__dict__, "prox_mu": mu, "noise_multiplier": DP_NM})


def main():
    os.makedirs("results", exist_ok=True)
    X, y = load_magic()
    stats = (f"MAGIC (OpenML 1120): {len(X)} samples, {X.shape[1]} features; "
             f"normal (gamma) {int((y == 0).sum())}, anomaly (hadron) {int((y == 1).sum())} "
             f"({100 * y.mean():.1f}% anomalous)")
    print(stats)
    open("results/dataset_stats.txt", "w").write(stats + "\n")

    done = set()
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = {(r["seed"], r["partition"], r["method"]) for r in rows}

    base = FLConfig(rounds=ROUNDS)
    for seed in SEEDS:
        splits = make_splits(X, y, seed)
        clients = {m: partition(splits.X_train_normal, base.n_clients, m, seed) for m in ("iid", "noniid")}
        for part, method, cfg in grid(base):
            if (seed, part, method) in done:
                continue
            t0 = time.time()
            if method == "centralized":
                res = run_centralized(splits, base, seed)
                nm, mu, eps = 0.0, 0.0, float("inf")
            elif method == "local-only":
                res = run_local_only(clients[part], splits, base, seed)
                nm, mu, eps = 0.0, 0.0, float("inf")
            else:
                res, _ = run_federated(clients[part], splits, cfg, seed)
                nm, mu, eps = cfg.noise_multiplier, cfg.prox_mu, res["epsilon"]
            rows.append({"seed": seed, "partition": part, "method": method, "noise_multiplier": nm,
                         "prox_mu": mu, "epsilon": eps, "auc": res["auc"], "auprc": res["auprc"],
                         "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
                         "best_round": res.get("best_round", np.nan)})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"seed={seed} {part:6s} {method:22s} AUC={res['auc']:.4f} "
                  f"eps={eps:.2f} ({time.time() - t0:.0f}s)", flush=True)
    if os.environ.get("SKIP_SUMMARY") != "1":
        summarize(pd.DataFrame(rows), stats)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def summarize(df, stats):
    n_seeds = df["seed"].nunique()
    order = {"none": 0, "iid": 1, "noniid": 2}
    g = df.groupby(["partition", "method"])
    agg = g.agg(auc_m=("auc", "mean"), auc_s=("auc", "std"), auprc_m=("auprc", "mean"),
                auprc_s=("auprc", "std"), f1_m=("f1", "mean"), f1_s=("f1", "std"),
                eps=("epsilon", "mean")).reset_index().fillna(0)
    agg["o"] = agg["partition"].map(order)
    agg = agg.sort_values(["o", "auc_m"], ascending=[True, False])

    lines = [f"# Results\n", f"{stats}\n",
             f"{n_seeds} seeds (mean ± std). Each seed changes the data split, the client partition and "
             f"the initialisation. Test labels are never used for model selection or the threshold; both "
             f"come from a separate validation set. ε is the total budget after all rounds at δ=1e-5 "
             f"(worst-case client, sample-level DP, RDP accountant).\n",
             "| Partition | Method | ROC AUC | AUPRC | F1 | ε |", "|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        eps = "–" if np.isinf(r["eps"]) else f"{r['eps']:.2f}"
        lines.append(f"| {r['partition']} | {r['method']} | {fmt(r['auc_m'], r['auc_s'])} | "
                     f"{fmt(r['auprc_m'], r['auprc_s'])} | {fmt(r['f1_m'], r['f1_s'])} | {eps} |")
    open("results/RESULTS.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # Chart 1: privacy vs utility (IID)
    iid = df[df["partition"] == "iid"]
    pts = []
    for nm in NOISE_SWEEP:
        d = iid[iid["method"] == f"fedavg+dp nm={nm}"]
        if len(d):
            pts.append((d["epsilon"].mean(), d["auc"].mean(), d["auc"].std() if len(d) > 1 else 0, nm))
    if pts:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        xs, ys, es, nms = zip(*sorted(pts))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="#2c7fb8", label="FedAvg + DP-SGD")
        for x, yv, nm in zip(xs, ys, nms):
            ax.annotate(f"σ={nm:g}", (x, yv), textcoords="offset points", xytext=(6, 6), fontsize=8)
        for name, color, ls in (("fedavg", "#c0392b", "--"), ("centralized", "#555", ":"), ("local-only", "#999", "-.")):
            d = df[(df["method"] == name) & (df["partition"].isin(["iid", "none"]))]
            if len(d):
                ax.axhline(d["auc"].mean(), color=color, ls=ls, lw=1.2,
                           label={"fedavg": "FedAvg, no DP", "centralized": "Centralized", "local-only": "Local only"}[name])
        ax.set_xlabel("total privacy budget ε (δ=1e-5, lower = more private)")
        ax.set_ylabel("test ROC AUC")
        ax.set_title(f"Privacy vs utility on MAGIC (IID, {n_seeds} seeds)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("results/privacy_utility.png", dpi=160)
        plt.close(fig)

    # Chart 2: FedProx on non-IID data, with and without DP
    ni = df[df["partition"] == "noniid"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width = 0.38
    for off, prefix, color, label in ((-width / 2, "fedavg mu=", "#2c7fb8", "no DP"),
                                      (width / 2, "fedavg+dp mu=", "#c0392b", f"DP (σ={DP_NM:g})")):
        m, s = [], []
        for mu in PROX_SWEEP:
            d = ni[ni["method"] == f"{prefix}{mu:g}"]["auc"]
            m.append(d.mean() if len(d) else np.nan)
            s.append(d.std() if len(d) > 1 else 0)
        ax.bar(np.arange(len(PROX_SWEEP)) + off, m, width, yerr=s, capsize=3, color=color, label=label)
    loc = ni[ni["method"] == "local-only"]["auc"]
    if len(loc):
        ax.axhline(loc.mean(), color="#999", ls="-.", lw=1.2, label="Local only")
    ax.set_xticks(range(len(PROX_SWEEP)))
    ax.set_xticklabels([f"μ={mu:g}" for mu in PROX_SWEEP])
    ax.set_ylabel("test ROC AUC")
    ax.set_title(f"FedProx on non-IID clients ({n_seeds} seeds)", fontsize=10)
    lo = np.nanmin([b.get_height() for b in ax.patches]) if ax.patches else 0.5
    ax.set_ylim(max(0.0, lo - 0.08), None)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/fedprox_noniid.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
